"""
Close the open canopy of a gmsh mesh with a "vent" surface.

Reads a gmsh 2.2 ASCII mesh, takes the triangles of the physical group "canopy", finds the open
boundary loops of that surface (edges used by a single triangle) and fills every loop with
triangles that only use the existing boundary nodes, so the vent is connected node-to-node to the
canopy and the canopy + vent surface is closed (watertight).

Outputs, next to the input mesh:
    <name>_of.msh       canopy (same physical name/ID as in the input) + "vent" physical group
    <name>_of_map.csv   original_triangle,of_triangle: gmsh element number of every canopy triangle
                        in the input mesh and in the _of mesh

The vent triangles have the same orientation as the canopy triangles next to them.

Each loop is triangulated by ear clipping on its projection onto the best-fit plane of the loop,
always cutting the ear with the largest minimum angle first (gives a clean strip for slit-like
openings such as the air intakes of a paraglider). The projected loop must not self-intersect.

Usage (from the tutorials directory):
    python close_canopy.py msh/demowing.msh
    python close_canopy.py msh/demowing.msh --canopy canopy --vent vent
"""
import argparse
import collections
import os

import numpy as np


# --------------------------------------------------------------------------------------------------
# gmsh 2.2 ASCII reading / writing
# --------------------------------------------------------------------------------------------------

def read_msh(path):
    """Return physical names {tag: (dim, name)}, nodes {id: xyz} and elements as a list of
    (id, type, tags, node ids)."""
    names, nodes, elements = {}, {}, []
    with open(path) as f:
        lines = iter(f.read().splitlines())
    for line in lines:
        if line.startswith("$MeshFormat"):
            version, filetype = next(lines).split()[:2]
            if not version.startswith("2") or filetype != "0":
                raise ValueError("%s: only gmsh 2.x ASCII meshes are supported (found %s)" % (path, version))
        elif line.startswith("$PhysicalNames"):
            for _ in range(int(next(lines))):
                dim, tag, name = next(lines).split(maxsplit=2)
                names[int(tag)] = (int(dim), name.strip().strip('"'))
        elif line.startswith("$Nodes"):
            for _ in range(int(next(lines))):
                a = next(lines).split()
                nodes[int(a[0])] = np.array([float(v) for v in a[1:4]])
        elif line.startswith("$Elements"):
            for _ in range(int(next(lines))):
                a = [int(v) for v in next(lines).split()]
                ntags = a[2]
                elements.append((a[0], a[1], a[3:3 + ntags], a[3 + ntags:]))
    return names, nodes, elements


def write_msh(path, names, nodes, elements):
    """names {tag: (dim, name)}, nodes [(id, xyz)], elements [(id, type, tags, node ids)]."""
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n%d\n" % len(names))
        for tag, (dim, name) in sorted(names.items()):
            f.write('%d %d "%s"\n' % (dim, tag, name))
        f.write("$EndPhysicalNames\n")
        f.write("$Nodes\n%d\n" % len(nodes))
        for i, x in nodes:
            f.write("%d %.12g %.12g %.12g\n" % (i, x[0], x[1], x[2]))
        f.write("$EndNodes\n")
        f.write("$Elements\n%d\n" % len(elements))
        for i, typ, tags, conn in elements:
            f.write(" ".join(str(v) for v in [i, typ, len(tags)] + list(tags) + list(conn)) + "\n")
        f.write("$EndElements\n")


# --------------------------------------------------------------------------------------------------
# open boundary of the canopy
# --------------------------------------------------------------------------------------------------

def boundary_loops(tri):
    """Closed loops of the edges used by a single triangle. Every loop is ordered opposite to the
    direction of its edges in the canopy triangles, so a triangle (loop[k-1], loop[k], loop[k+1])
    has the same orientation as the canopy next to it.

    Several loops may touch at a single node (pinch node, e.g. neighbouring air intakes meeting at
    a rib). There the canopy triangles around the node form separate fans, and a loop arriving
    along the boundary edge of one fan leaves along the boundary edge of the other fan."""
    edge_tri = collections.defaultdict(list)       # undirected edge -> triangles using it
    directed = set()
    for k, t in enumerate(tri):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_tri[frozenset((a, b))].append(k)
            directed.add((a, b))

    nonmanifold = [e for e, ts in edge_tri.items() if len(ts) > 2]
    if nonmanifold:
        raise ValueError("canopy has %d edges shared by more than 2 triangles" % len(nonmanifold))

    # boundary edges in loop direction (against the canopy edge direction: b -> a)
    out_edges = collections.defaultdict(list)
    for e, ts in edge_tri.items():
        if len(ts) == 1:
            a, b = tuple(e)
            if (a, b) not in directed:
                a, b = b, a                         # canopy has the directed edge a -> b
            out_edges[b].append(a)

    def fan_of(v, w):
        """Fan (set of triangles around v connected through interior edges) containing the
        triangle of the boundary edge v-w."""
        start = edge_tri[frozenset((v, w))][0]
        fan, stack = {start}, [start]
        while stack:
            t = tri[stack.pop()]
            for u in t:
                if u != v:
                    for k in edge_tri[frozenset((v, u))]:
                        if k not in fan:
                            fan.add(k)
                            stack.append(k)
        return frozenset(fan)

    def next_node(prev, v):
        candidates = out_edges[v]
        if len(candidates) == 1:
            return candidates[0]
        arriving_fan = fan_of(v, prev)
        others = [w for w in candidates if fan_of(v, w) != arriving_fan]
        if len(others) != 1:
            raise ValueError("cannot follow the canopy boundary through node %d (%d boundary loops "
                             "meet there)" % (v, len(candidates)))
        return others[0]

    loops, used = [], set()                         # used: directed loop edges already walked
    for start in out_edges:
        for first in out_edges[start]:
            if (start, first) in used:
                continue
            loop, prev, node = [start], start, first
            used.add((start, first))
            while node != start:
                loop.append(node)
                prev, node = node, next_node(prev, node)
                if (prev, node) in used:
                    raise ValueError("open boundary chain found, the canopy boundary is not a set of loops")
                used.add((prev, node))
            loops.append(loop)
    return loops


# --------------------------------------------------------------------------------------------------
# loop triangulation
# --------------------------------------------------------------------------------------------------

def segments_intersect(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def check_simple(Q):
    """Raise if the projected polygon Q (n x 2) intersects itself."""
    n = len(Q)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if segments_intersect(Q[i], Q[(i + 1) % n], Q[j], Q[(j + 1) % n]):
                raise ValueError("the loop projected on its best-fit plane intersects itself "
                                 "(edges %d and %d), cannot triangulate it" % (i, j))


def min_angle(a, b, c):
    """Smallest angle of triangle abc in 3D [rad]."""
    angles = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        u, v = q - p, r - p
        cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        angles.append(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return min(angles)


def triangulate_loop(loop, X):
    """Triangles (node ids) filling the closed loop, oriented like the loop order."""
    P = np.array([X[i] for i in loop])
    centre = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - centre)
    Q = (P - centre) @ vt[:2].T

    # ear clipping needs a counter-clockwise polygon: mirror the projection, not the loop order,
    # so the triangles keep the orientation of the loop
    area = 0.5 * np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1])
    if area < 0:
        Q[:, 1] = -Q[:, 1]
    check_simple(Q)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) > 0 and cross(b, c, p) > 0 and cross(c, a, p) > 0

    idx = list(range(len(loop)))

    def ear_quality(k):
        """Minimum angle of the ear at position k, or -1 if it is not a valid ear."""
        i, j, l = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
        a, b, c = Q[i], Q[j], Q[l]
        if cross(a, b, c) <= 0:                     # reflex or degenerate corner
            return -1.0
        for m in idx:
            if m not in (i, j, l) and inside(Q[m], a, b, c):
                return -1.0
        return min_angle(P[i], P[j], P[l])

    quality = {idx[k]: ear_quality(k) for k in range(len(idx))}
    triangles = []
    while len(idx) > 3:
        k = max(range(len(idx)), key=lambda k: quality[idx[k]])
        if quality[idx[k]] < 0:
            raise ValueError("no valid ear left, cannot triangulate the loop")
        i, j, l = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
        triangles.append((loop[i], loop[j], loop[l]))
        del idx[k]
        del quality[j]
        k_prev = (k - 1) % len(idx)                 # neighbours of the removed corner
        for kk in (k_prev, (k_prev + 1) % len(idx)):
            quality[idx[kk]] = ear_quality(kk)
    triangles.append(tuple(loop[i] for i in idx))
    return triangles


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------

def close_canopy(mesh, canopy_group="canopy", vent_group="vent"):
    """Write <mesh>_of.msh and <mesh>_of_map.csv next to mesh, return their paths."""
    names, nodes, elements = read_msh(mesh)
    canopy_tag = [tag for tag, (dim, name) in names.items() if dim == 2 and name == canopy_group]
    if not canopy_tag:
        raise ValueError("no 2D physical group '%s' in %s" % (canopy_group, mesh))
    canopy_tag = canopy_tag[0]

    canopy = [e for e in elements if e[2] and e[2][0] == canopy_tag]
    if any(e[1] != 2 for e in canopy):
        raise ValueError("physical group '%s' contains non-triangle elements" % canopy_group)
    tri = [e[3] for e in canopy]
    print("%s: %d canopy triangles" % (mesh, len(tri)))

    # close every open loop
    loops = boundary_loops(tri)
    vent_tri = []
    for k, loop in enumerate(loops):
        try:
            vent_tri.extend(triangulate_loop(loop, nodes))
        except ValueError as err:
            raise ValueError("open loop %d (%d nodes, first node %d): %s" % (k, len(loop), loop[0], err))
    sizes = collections.Counter(len(loop) for loop in loops)
    angles = [np.degrees(min_angle(*(nodes[i] for i in t))) for t in vent_tri]
    print("  %d open loops (%s) closed with %d vent triangles"
          % (len(loops), ", ".join("%d x %d nodes" % (c, n) for n, c in sorted(sizes.items())), len(vent_tri)))
    print("  vent triangle min angle: smallest %.1f deg, mean %.1f deg" % (min(angles), np.mean(angles)))

    # canopy + vent must be closed: every edge in exactly two triangles
    count = collections.Counter(frozenset(e) for t in tri + vent_tri
                                for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
    bad = sum(1 for c in count.values() if c != 2)
    print("  closed surface check: %s" % ("OK" if bad == 0 else "%d edges not shared by 2 triangles" % bad))

    # renumber nodes (canopy nodes only, in original order) and elements
    used = sorted({i for t in tri for i in t})
    new_node = {old: new for new, old in enumerate(used, start=1)}
    vent_tag = max(names) + 1
    vent_entity = max(t for e in elements for t in e[2][1:2]) + 1 if any(len(e[2]) > 1 for e in elements) else vent_tag

    out_elements, mapping = [], []
    for k, (eid, typ, tags, conn) in enumerate(canopy, start=1):
        out_elements.append((k, 2, tags, [new_node[i] for i in conn]))
        mapping.append((eid, k))
    for t in vent_tri:
        out_elements.append((len(out_elements) + 1, 2, [vent_tag, vent_entity], [new_node[i] for i in t]))

    base = os.path.splitext(mesh)[0]
    out_msh, out_csv = base + "_of.msh", base + "_of_map.csv"
    write_msh(out_msh, {canopy_tag: (2, canopy_group), vent_tag: (2, vent_group)},
              [(new_node[i], nodes[i]) for i in used], out_elements)
    np.savetxt(out_csv, np.array(mapping, dtype=int), fmt="%d", delimiter=",",
               header="original_triangle,of_triangle", comments="")

    print("written %s: %d nodes, %d canopy + %d vent triangles"
          % (out_msh, len(used), len(canopy), len(vent_tri)))
    print("written %s" % out_csv)
    return out_msh, out_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", help="gmsh 2.2 ASCII mesh, e.g. msh/demowing.msh")
    parser.add_argument("--canopy", default="canopy", help="physical group of the canopy (default: canopy)")
    parser.add_argument("--vent", default="vent", help="physical group of the closing surface (default: vent)")
    args = parser.parse_args()
    close_canopy(args.mesh, args.canopy, args.vent)


if __name__ == "__main__":
    main()
