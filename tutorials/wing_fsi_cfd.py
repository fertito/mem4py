"""
OpenFOAM run of the last iteration of a wing directory, and canopy pressure CSV from its result.

A wing directory <wing>/ (default demowing/) contains <wing>.msh and <wing>.json, whose
"iterations" list is extended by paraglider_fsi.py after each structural solve. This script takes
the last iteration NN of the json (or --iteration NN) and runs the flow around the wing with
wing_openfoam.py:

    speed           = Speed                       [m/s]
    angle of attack = Glide_angle                 [deg], free stream onto the wing in its mesh position

Wing position: before the CFD, the wing is rotated by Wing_Roll, Wing_Pitch and Wing_Yaw [deg]
(right-handed rotations about the mesh axes x chordwise, y spanwise, z up, applied roll first, then
pitch, then yaw: R = Rz(yaw) Ry(pitch) Rx(roll)) around the pivot, the middle of the main_l and
main_r nodes of <wing>_attachments.csv (the origin if that file is missing). With no roll and yaw the
angle of attack seen by the wing is Glide_angle + Wing_Pitch.

Wing shape:
    - iteration 0: the undeformed canopy of <wing>.msh
    - iteration NN > 0: the canopy deformed by the structural solve of the previous iteration,
      read from <wing>/iter_{NN-1}/<wing>_canopy.vtk (written by paraglider_fsi.py --solve or
      --export); use --undeformed to take <wing>.msh instead.

Everything is written to <wing>/iter_NN/ (created if needed):
    <wing>_cfd.msh          <wing>.msh with the canopy nodes at the deformed position, the whole
                            wing rotated by Wing_Roll, Wing_Pitch, Wing_Yaw
    <wing>_cfd_of.msh       canopy closed with a vent surface (close_canopy.py)
    <wing>_cfd_of_map.csv   gmsh triangle number in <wing>.msh -> in <wing>_cfd_of.msh
    openfoam/               OpenFOAM case, open openfoam/case.foam in ParaView

Canopy pressure: once OpenFOAM has run, the latest time of <wing>/iter_NN/openfoam is read and the
pressure of every canopy triangle is written to <wing>/<wing>_pressure.csv, the file read by
paraglider_fsi.py:

    triangle,pressure
    7367,18.194801
    ...

"triangle" is the gmsh element number of the canopy triangle in <wing>.msh and "pressure" the
static gauge pressure in Pa, p = RHO_AIR * p_kinematic (the free stream pressure of the case is 0).

Mapping: the pressure of a wing_canopy face is the value of its wall cell (zeroGradient patch). Each
canopy triangle of <wing>/iter_NN/<wing>_cfd.msh (the wing shape used by the CFD) takes the
area / distance weighted mean of the nearest wing_canopy faces whose normal points the same way as
the triangle, so the faces of the opposite skin near the trailing edge are not mixed in.

Vent pressure: the maximum face pressure of the wing_vent patch times P_vent_ratio of the iteration
NN (1.0 if missing), P_vent = P_vent_ratio * max(p_i) [Pa], replaces P_vent of the iteration NN in
<wing>.json (internal pressure used by paraglider_fsi.py --iteration NN).

The previous <wing>_pressure.csv is kept as <wing>/iter_NN/<wing>_pressure_previous.csv and a copy
of the new one is written to <wing>/iter_NN/<wing>_pressure.csv.

Usage (from the tutorials directory):
    python wing_fsi_cfd.py                       last iteration of demowing/demowing.json
    python wing_fsi_cfd.py --wing mywing --iteration 0
    python wing_fsi_cfd.py --iterations 2000 --procs 8
    python wing_fsi_cfd.py --no-run              only write the meshes and the case
    python wing_fsi_cfd.py --pressure-only       only write the pressure CSV of an existing result
    then: python paraglider_fsi.py --iteration NN --solve
"""
import argparse
import csv
import itertools
import json
import os
import re
import shutil

import numpy as np
from scipy.spatial import cKDTree

from close_canopy import read_msh, write_msh, close_canopy
from paraglider_fsi import read_mem4py_vtk
from wing_openfoam import run_wing, ITERATIONS, RHO_AIR


WING_DIR = "demowing"

K_NEAREST = 16          # candidate faces per triangle
MIN_NORMAL_DOT = 0.3    # cosine between triangle and face normals to accept a face


def json_iteration(json_path, iteration=None):
    """Iteration dict of the json, the last one when iteration is None."""
    with open(json_path) as f:
        iterations = json.load(f)["iterations"]
    if not iterations:
        raise ValueError("no iteration in %s" % json_path)
    if iteration is None:
        return max(iterations, key=lambda it: it["iteration"])
    for it in iterations:
        if it["iteration"] == iteration:
            return it
    raise KeyError("iteration %d not found in %s" % (iteration, json_path))


# --------------------------------------------------------------------------------------------------
# wing shape
# --------------------------------------------------------------------------------------------------

def deformed_nodes(nodes, elements, canopy_vtk):
    """Copy of the msh nodes {id: xyz} with the canopy nodes moved to their deformed position.

    The canopy vtk of paraglider_fsi.py stores the deformed points, the displacement u and, per
    triangle, its gmsh element number ("triangle"). The vtk nodes of a triangle are matched to the
    gmsh nodes of that element by their undeformed position (points - u), so the node ordering of
    the vtk does not matter."""
    vtk = read_mem4py_vtk(canopy_vtk)
    deformed = vtk["points"]
    undeformed = deformed - vtk["point_data"]["u"][2]
    triangle = vtk["cell_data"]["triangle"][2].astype(int)
    conn = {e[0]: e[3] for e in elements if e[1] == 2}

    moved, worst = {}, 0.0
    for k, eid in enumerate(triangle):
        if eid not in conn:
            raise ValueError("triangle %d of %s is not in the mesh" % (eid, canopy_vtk))
        for v in vtk["cells"][k]:
            ids = conn[eid]
            dist = [np.linalg.norm(nodes[i] - undeformed[v]) for i in ids]
            j = int(np.argmin(dist))
            worst = max(worst, dist[j])
            moved[ids[j]] = deformed[v]
    print("  %s: %d canopy nodes moved (max position mismatch %.2g m)" % (canopy_vtk, len(moved), worst))
    if worst > 1E-6:
        raise ValueError("vtk nodes of %s do not match the mesh nodes (mismatch %.3g m)" % (canopy_vtk, worst))

    out = dict(nodes)
    out.update(moved)
    return out


def rotation(roll=0.0, pitch=0.0, yaw=0.0):
    """R = Rz(yaw) Ry(pitch) Rx(roll), angles in degrees, right-handed about the mesh axes."""
    r, p, y = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_pivot(wing_dir, name, nodes):
    """Middle of the main_l and main_r nodes of <wing>_attachments.csv, the origin if missing."""
    path = os.path.join(wing_dir, name + "_attachments.csv")
    if not os.path.isfile(path):
        print("  %s not found, wing rotated about the origin" % path)
        return np.zeros(3)
    with open(path, newline="") as f:
        att = {r["name"].strip(): int(r["node"]) for r in csv.DictReader(f)}
    mains = [n for n in ("main_l", "main_r") if n in att]
    if not mains:
        print("  no main_l / main_r in %s, wing rotated about the origin" % path)
        return np.zeros(3)
    return np.mean([nodes[att[n]] for n in mains], axis=0)


def rotate_wing(nodes, roll, pitch, yaw, pivot):
    """Copy of the msh nodes {id: xyz} rotated by R = rotation(roll, pitch, yaw) about pivot."""
    R = rotation(roll, pitch, yaw)
    print("  wing rotated by roll %g, pitch %g, yaw %g deg about (%.4g, %.4g, %.4g)"
          % ((roll, pitch, yaw) + tuple(pivot)))
    return {i: pivot + R @ (np.asarray(x) - pivot) for i, x in nodes.items()}


# --------------------------------------------------------------------------------------------------
# OpenFOAM ascii reading
# --------------------------------------------------------------------------------------------------

def _list_lines(path, marker=None):
    """Lines of the first ascii list of an OpenFOAM file (after marker if given)."""
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    if marker:
        while marker not in lines[i]:
            i += 1
    while not lines[i].strip().isdigit():
        i += 1
    n = int(lines[i])
    return lines[i + 2:i + 2 + n]


FACE_RE = re.compile(r"(\d+)\s*\(([^)]*)\)")


def read_faces(path, start, n):
    """Vertex index arrays of faces start .. start + n - 1 of constant/polyMesh/faces.

    OpenFOAM writes a face as "4(a b c d)" on one line, but a face of more than 10 vertices as
        11
        (
        a
        ...
        )
    over several lines, so the faces are read as "count ( indices )" tokens, whatever the layout."""
    with open(path) as f:
        text = f.read()
    m = re.search(r"^\s*(\d+)\s*\n\s*\(", text, re.M)     # list size, then the opening "("
    if not m:
        raise ValueError("no face list in %s" % path)
    faces = []
    for k, fm in enumerate(itertools.islice(FACE_RE.finditer(text, m.end()), start, start + n)):
        ids = np.array(fm.group(2).split(), dtype=int)
        if len(ids) != int(fm.group(1)):
            raise ValueError("face %d of %s: %s vertices announced, %d read" % (start + k, path, fm.group(1), len(ids)))
        faces.append(ids)
    if len(faces) != n:
        raise ValueError("%s: %d faces read from %d, %d expected" % (path, len(faces), start, n))
    return faces


def read_boundary(path):
    """{patch: (startFace, nFaces)} of constant/polyMesh/boundary."""
    with open(path) as f:
        text = f.read()
    patches, name = {}, None
    for line in text.split("\n"):
        tokens = line.replace(";", "").split()
        if len(tokens) == 1 and tokens[0].startswith("wing_") or tokens == ["farfield"]:
            name = tokens[0]
            patches[name] = {}
        elif name and len(tokens) == 2 and tokens[0] in ("nFaces", "startFace"):
            patches[name][tokens[0]] = int(tokens[1])
    return {k: (v["startFace"], v["nFaces"]) for k, v in patches.items()}


def latest_time(case):
    times = [d for d in os.listdir(case) if os.path.isdir(os.path.join(case, d))
             and d.replace(".", "", 1).isdigit() and float(d) > 0]
    if not times:
        raise FileNotFoundError("no result time directory in %s (run wing_fsi_cfd.py first)" % case)
    return max(times, key=float)


def wall_faces(case, time, patch="wing_canopy"):
    """Centres, area vectors (pointing into the wing) and kinematic pressure of the patch faces."""
    mesh = os.path.join(case, "constant", "polyMesh")
    start, n = read_boundary(os.path.join(mesh, "boundary"))[patch]

    points = np.array([l.strip("()").split() for l in _list_lines(os.path.join(mesh, "points"))], dtype=float)
    faces = read_faces(os.path.join(mesh, "faces"), start, n)
    owner = np.array(_list_lines(os.path.join(mesh, "owner"))[start:start + n], dtype=int)
    p_cell = np.array(_list_lines(os.path.join(case, time, "p"), "internalField"), dtype=float)

    centre, area = np.empty((n, 3)), np.empty((n, 3))
    for k, ids in enumerate(faces):
        X = points[ids]
        c = X.mean(axis=0)
        a = 0.5 * np.cross(X - c, np.roll(X, -1, axis=0) - c).sum(axis=0)
        tri_area = np.linalg.norm(np.cross(X - c, np.roll(X, -1, axis=0) - c), axis=1)
        centre[k] = (((X + np.roll(X, -1, axis=0) + c) / 3) * tri_area[:, None]).sum(axis=0) / tri_area.sum()
        area[k] = a
    return centre, area, p_cell[owner]


# --------------------------------------------------------------------------------------------------
# pressure mapping
# --------------------------------------------------------------------------------------------------

def canopy_triangles(msh):
    """gmsh element numbers, centroids and area vectors of the canopy triangles of msh."""
    names, nodes, elements = read_msh(msh)
    tag = [t for t, (dim, name) in names.items() if dim == 2 and name == "canopy"][0]
    canopy = [e for e in elements if e[1] == 2 and e[2][0] == tag]
    X = np.array([[nodes[i] for i in e[3]] for e in canopy])
    ids = np.array([e[0] for e in canopy], dtype=int)
    return ids, X.mean(axis=1), 0.5 * np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])


def map_pressure(tri_c, tri_a, face_c, face_a, face_p):
    tri_n = tri_a / np.linalg.norm(tri_a, axis=1)[:, None]
    face_area = np.linalg.norm(face_a, axis=1)
    face_n = face_a / face_area[:, None]
    tree = cKDTree(face_c)

    # orientation of the canopy triangles relative to the patch normals (one sign for the whole mesh)
    _, nearest = tree.query(tri_c)
    sign = np.sign(np.median(np.einsum("ij,ij->i", tri_n, face_n[nearest])))
    tri_n = sign * tri_n

    dist, idx = tree.query(tri_c, k=K_NEAREST)
    dot = np.einsum("ij,ikj->ik", tri_n, face_n[idx])
    h = np.sqrt(face_area[idx])
    w = np.where(dot > MIN_NORMAL_DOT, face_area[idx] / (dist ** 2 + h ** 2), 0.0)

    # no compatible face among the nearest: widen the search, and if there is still none (canopy
    # detail not resolved by the CFD mesh, e.g. inside a narrow intake) take the nearest face
    missing = np.flatnonzero(w.sum(axis=1) == 0)
    nearest_only = []
    for k in missing:
        d, j = tree.query(tri_c[k], k=20 * K_NEAREST)
        ok = np.flatnonzero(face_n[j] @ tri_n[k] > MIN_NORMAL_DOT)
        if len(ok) == 0:
            nearest_only.append(k)
        idx[k, 0], w[k] = j[ok[0] if len(ok) else 0], 0.0
        w[k, 0] = 1.0
    p = (w * face_p[idx]).sum(axis=1) / w.sum(axis=1)
    return p, sign, len(missing), np.array(nearest_only, dtype=int)


def write_pressure_csv(wing_dir, name, out_dir):
    """Canopy pressure of the OpenFOAM result in out_dir/openfoam -> <wing>/<wing>_pressure.csv."""
    case = os.path.join(out_dir, "openfoam")
    time = latest_time(case)
    print("canopy pressure: OpenFOAM result %s, time %s" % (case, time))

    face_c, face_a, face_p = wall_faces(case, time)
    face_p = RHO_AIR * face_p
    print("  wing_canopy: %d faces, p from %.1f to %.1f Pa" % (len(face_p), face_p.min(), face_p.max()))

    ids, tri_c, tri_a = canopy_triangles(os.path.join(out_dir, name + "_cfd.msh"))
    p, sign, widened, nearest_only = map_pressure(tri_c, tri_a, face_c, face_a, face_p)
    print("  %d canopy triangles mapped (normals %s the patch normals, %d by widened search), "
          "p from %.1f to %.1f Pa" % (len(ids), "along" if sign > 0 else "opposite to", widened, p.min(), p.max()))
    if len(nearest_only):
        print("  Warning: %d triangles without a CFD face of compatible normal nearby (first: %d at %s), "
              "nearest face used" % (len(nearest_only), ids[nearest_only[0]], np.round(tri_c[nearest_only[0]], 3)))

    # check: pressure force on the CFD faces and on the triangles (both along the patch normals)
    F_faces = (face_p[:, None] * face_a).sum(axis=0)
    F_tri = (p[:, None] * sign * tri_a).sum(axis=0)
    print("  pressure force on the canopy [N]: CFD faces (%.1f, %.1f, %.1f), triangles (%.1f, %.1f, %.1f)"
          % (tuple(F_faces) + tuple(F_tri)))

    csv = os.path.join(wing_dir, name + "_pressure.csv")
    if os.path.isfile(csv):
        shutil.copyfile(csv, os.path.join(out_dir, name + "_pressure_previous.csv"))
    np.savetxt(csv, np.column_stack((ids, p)), fmt=["%d", "%.6f"], delimiter=",",
               header="triangle,pressure", comments="")
    shutil.copyfile(csv, os.path.join(out_dir, name + "_pressure.csv"))
    print("written %s (copy in %s, previous file kept as %s_pressure_previous.csv)" % (csv, out_dir, name))
    return csv


def update_vent_pressure(wing_dir, name, out_dir, iteration):
    """Maximum face pressure of the wing_vent patch of the latest OpenFOAM time, scaled by
    P_vent_ratio of the given iteration of <wing>.json (1.0 if the key is missing),
        P_vent = P_vent_ratio * max(p_i)   [Pa],  p = RHO_AIR * p_kinematic,
    written as P_vent of that iteration (the previous value is printed)."""
    json_path = os.path.join(wing_dir, name + ".json")
    with open(json_path) as f:
        doc = json.load(f)
    it = next(it for it in doc["iterations"] if it["iteration"] == iteration)
    vent_ratio = float(it.get("P_vent_ratio", 1.0))

    case = os.path.join(out_dir, "openfoam")
    time = latest_time(case)
    _, face_a, face_p = wall_faces(case, time, "wing_vent")
    area = np.linalg.norm(face_a, axis=1)
    p = RHO_AIR * face_p
    p_mean = float((p * area).sum() / area.sum())
    p_vent = vent_ratio * float(p.max())
    print("vent pressure: %d wing_vent faces, %.4g m2, p from %.1f to %.1f Pa (area weighted mean %.2f Pa), "
          "P_vent = %g * max = %.2f Pa"
          % (len(p), area.sum(), p.min(), p.max(), p_mean, vent_ratio, p_vent))

    old = it.get("P_vent")
    it["P_vent"] = p_vent
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=4)
        f.write("\n")
    print("  %s iteration %d: P_vent %s -> %.2f Pa" % (json_path, iteration, old, p_vent))
    return p_vent


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wing", default=WING_DIR, help="wing directory (default: %s)" % WING_DIR)
    parser.add_argument("--iteration", type=int, help="json iteration (default: the last one)")
    parser.add_argument("--undeformed", action="store_true", help="use the undeformed canopy of <wing>.msh")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="SIMPLE iterations (default %d)" % ITERATIONS)
    parser.add_argument("--procs", type=int, default=16, help="MPI processes (default 16, 1 = serial)")
    parser.add_argument("--no-run", action="store_true", help="only write the meshes and the case")
    parser.add_argument("--pressure-only", action="store_true",
                        help="only write the pressure CSV of the existing OpenFOAM result")
    args = parser.parse_args()

    wing_dir = os.path.normpath(args.wing)
    name = os.path.basename(wing_dir)
    msh = os.path.join(wing_dir, name + ".msh")
    it = json_iteration(os.path.join(wing_dir, name + ".json"), args.iteration)
    n = it["iteration"]
    speed = float(it["Speed"])
    glide = float(it["Glide_angle"])
    roll, pitch, yaw = (float(it.get(k, 0.0)) for k in ("Wing_Roll", "Wing_Pitch", "Wing_Yaw"))
    out_dir = os.path.join(wing_dir, "iter_%02d" % n)
    os.makedirs(out_dir, exist_ok=True)
    print("%s iteration %d: speed %g m/s, Glide_angle %g deg, wing roll %g, pitch %g, yaw %g deg"
          % (name, n, speed, glide, roll, pitch, yaw))

    if not args.pressure_only:
        # wing shape
        names, nodes, elements = read_msh(msh)
        if n > 0 and not args.undeformed:
            canopy_vtk = os.path.join(wing_dir, "iter_%02d" % (n - 1), name + "_canopy.vtk")
            if not os.path.isfile(canopy_vtk):
                raise FileNotFoundError("%s not found: run paraglider_fsi.py --iteration %d --solve (or --export) "
                                        "first, or use --undeformed" % (canopy_vtk, n - 1))
            nodes = deformed_nodes(nodes, elements, canopy_vtk)
        else:
            print("  undeformed canopy of %s" % msh)
        if roll or pitch or yaw:
            nodes = rotate_wing(nodes, roll, pitch, yaw, rotation_pivot(wing_dir, name, nodes))

        cfd_msh = os.path.join(out_dir, name + "_cfd.msh")
        write_msh(cfd_msh, names, sorted(nodes.items()), elements)
        of_msh, _ = close_canopy(cfd_msh)

        run_wing(of_msh, glide, speed, os.path.join(out_dir, "openfoam"), args.iterations, args.procs,
                 run=not args.no_run)

    if not args.no_run:
        write_pressure_csv(wing_dir, name, out_dir)
        update_vent_pressure(wing_dir, name, out_dir, n)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
