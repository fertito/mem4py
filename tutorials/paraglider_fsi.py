"""
Paraglider loaded with an FSI pressure field.

Every wing lives in its own directory <wing>/ (default demowing/) containing

    <wing>.msh              gmsh mesh, the canopy is declared as an FSI surface in mem4py
    <wing>_pressure.csv     pressure of every canopy triangle
    <wing>.json             run parameters, "iterations"[0] gives
                                P_ratio  multiplier applied to the CSV pressure
                                P_vent   internal pressure [Pa] added to every canopy triangle
    <wing>_canopy.csv       triangles (header "triangle,piece") or lines (header "line,piece")
    <wing>_diagonals.csv    of each part of the wing, as gmsh element numbers, used to split the
    <wing>_ribs.csv         vtk result per part (optional)
    <wing>_straps.csv
    <wing>_bridle.csv
    <wing>_sew.csv          if missing, generated from the "sew" physical group of <wing>.msh,
                            piece = "sew:<gmsh entity tag>"
    <wing>_attachments.csv  attachment points (node = msh node, name, type), main_l / main_r are
                            the left / right pilot attachment points
    <wing>_bridle_material.csv  optional, material of every bridle line (header
                            id,name,linetype,diameter,E_secant; id = gmsh geometrical entity of the
                            line, diameter [m], E_secant [Pa]): replaces d and E of elStruc["bridle"]
                            for the segments of that line, see apply_bridle_material

Line drag: every cable element of the "bridle" physical group gets the cross-flow drag of a circular
cylinder, F = 1/2 RHO_AIR LINE_CD d L |U_n| U_n (LINE_CD = 1.2, U_n = free stream normal to the
segment, U = Speed at Glide_angle, rotated back by Wing_Roll / Wing_Pitch / Wing_Yaw into the mesh
frame), added to the forces calculated by mem4py, see apply_line_drag.

The pressure CSV looks like

    triangle,pressure
    7367,18.194801
    ...

where "triangle" is the gmsh element number in <wing>.msh and "pressure" the value applied to that
triangle; it is multiplied by P_ratio and its sign is inverted on reading
(csm.pFSI = -P_ratio * pressure + P_vent). The CSV triangles are matched to the mem4py membrane
elements via their node connectivity, so the mapping does not depend on the element ordering inside
mem4py.

All output goes to <wing>/iter_NN/, NN being the json iteration: a picture of the paraglider
coloured by the pressure (paraglider_pressure.png) and, with --solve, the mem4py vtk result, the
picture of the deformed wing (paraglider_pressure_deformed.png) and a picture at every kinetic
energy peak in <wing>/iter_NN/peaks/.

After the solve, the mem4py vtk result is split into one vtk per part CSV, <wing>_canopy.vtk,
<wing>_diagonals.vtk, <wing>_ribs.vtk, <wing>_straps.vtk, <wing>_bridle.vtk and <wing>_sew.vtk, containing only
the triangles or lines of that CSV (in CSV order) at their deformed position (undeformed points + u),
with every point field of the mem4py vtk (u, ...), its membrane cell fields (triangles) or cable
cell fields (lines), and the cell field "triangle" or "line", the gmsh element number of each
element in <wing>.msh.

<wing>_force.vtk shows the pressure force on the canopy: for every triangle of <wing>_canopy.csv a
line from the triangle centre, along its normal, |F| = |p| * area long (1 N = 1 m), with the cell
field force = p * area [N] (p = pressure applied in mem4py), see write_force_vtk.
<wing>_force_sum.vtk holds a single line, the resultant R of all these forces, drawn from a point of
its line of action and FORCE_SUM_SCALE * |R| long (0.005 m/N), see write_force_sum_vtk.
<wing>_attachments_force.csv gives, for every node of <wing>_attachments.csv, the pull of the
bridle lines on it and the mem4py reaction force, see write_attachment_forces.

Finally the next iteration is added to <wing>.json (replaced if it already exists):
    P_ratio_next = NEXT_P_RATIO (1.0),  P_vent_next = NEXT_P_VENT (50.0)
    Force_main_left_next / Force_main_right_next = magnitude of the reaction force F_reaction [N]
                      of the mem4py vtk result at the nodes main_l / main_r of <wing>_attachments.csv
    Wing_Pitch_next = rotation [deg] in the side (x-z) plane during the solve of the main risers
                      listed in <wing>_data.csv (main_riser <piece> main_node / upper_node, msh
                      nodes), averaged over all of them; positive when the upper node moves
                      toward +x relative to the main node.
The other keys (Wing_Roll, Wing_Yaw, Speed, Glide_angle, ...) are copied unchanged.

Usage (from the tutorials directory, with the mem4py venv):
    python paraglider_fsi.py                map the pressure and plot it on the undeformed wing
    python paraglider_fsi.py --solve        additionally solve the structure and plot the deformed wing
                                            (slow: the demowing case takes well over an hour)
    python paraglider_fsi.py --wing mywing  use the wing in directory mywing/
    python paraglider_fsi.py --iteration 1  use P_ratio and P_vent of json iteration 1, output in iter_01/
    python paraglider_fsi.py --ratio 0.5    override P_ratio of the json (combinable with --solve)
    python paraglider_fsi.py --export       no solve: split the existing vtk result of iter_NN/ per
                                            part and prepare the next iteration of the json
    python paraglider_fsi.py --solve --pretension
                                            pre-tension the lines of <wing>/<wing>_control.csv
    python paraglider_fsi.py --solve --pretension mylines.csv
                                            pre-tension the lines of another CSV

Line pre-tension (--pretension): the CSV gives, per line, the change of its length in percent

    "name","length","id","stretch"
    "main left",0.540094,85,1.5
    ...

"id" is the gmsh geometrical entity of the line in <wing>.msh (second element tag; the line is made
of all the 2-node elements of that entity), "stretch" [%] the change of the line length relative
to the mesh (negative = shorter line, pre-tensioned at start, e.g. -50 = 50 % shorter; positive =
longer line, slack at start). The rest length of every segment of the line is set to
L_mesh * (1 + stretch / 100) before the solve, so a shortened line starts stretched and pulls toward
its equilibrium. "name" (optional) is only used in the printout and
"length" (optional) is checked against the mesh length of the whole line.
"""
import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import mem4py as m4p


WING_DIR = "demowing"            # default wing directory, see --wing
JSON_ITERATION = 0               # default json iteration for P_ratio, P_vent and the output directory, see --iteration
PARTS = ("canopy", "diagonals", "ribs", "straps", "bridle", "sew")    # <wing>_<part>.csv -> <wing>_<part>.vtk
PHYSICAL_PARTS = ("sew",)        # part CSVs generated from the gmsh physical group of that name if missing
NEXT_P_RATIO = 1.0               # P_ratio written to the next iteration of the json
NEXT_P_VENT = 50.0               # [Pa] P_vent written to the next iteration of the json
FORCE_SUM_SCALE = 0.005          # [m/N] length of the resultant line of <wing>_force_sum.vtk
LINE_CD = 1.2                    # drag coefficient of a circular cylinder in cross flow (subcritical Re)
LINE_DRAG_SETS = ("bridle",)     # physical groups of cable elements loaded by the line drag
RHO_AIR = 1.225                  # [kg/m3] air density of the line drag (same as the CFD)


class Wing:
    """Files of a wing directory: <dir>/<name>.msh, <dir>/<name>_pressure.csv, <dir>/<name>.json,
    with <name> the directory name."""

    def __init__(self, directory):
        self.dir = os.path.normpath(directory)
        self.name = os.path.basename(self.dir)
        self.msh = os.path.join(self.dir, self.name + ".msh")
        self.pressure_csv = os.path.join(self.dir, self.name + "_pressure.csv")
        self.json = os.path.join(self.dir, self.name + ".json")
        for path in (self.msh, self.pressure_csv, self.json):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

    def iteration(self, iteration=JSON_ITERATION):
        """The json entry of the given iteration (dict)."""
        with open(self.json) as f:
            iterations = json.load(f)["iterations"]
        for it in iterations:
            if it["iteration"] == iteration:
                return it
        raise KeyError("iteration %d not found in %s" % (iteration, self.json))

    def iteration_parameters(self, iteration=JSON_ITERATION):
        """(P_ratio, P_vent) of the given iteration in the json."""
        it = self.iteration(iteration)
        return float(it["P_ratio"]), float(it["P_vent"])


# --------------------------------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------------------------------

def paraglider_problem(wing, output_dir):
    """Paraglider of main.py, with the canopy loaded by per-element FSI pressure instead of a
    uniform pressure."""

    elStruc = {}
    elStruc["fix"] = {"set": "BC",
                      "type": "fixAll"}

    # elStruc["fixXY"] = {"set": "BC",
    #                     "type": "fixXY"}

    elStruc["canopy"] = {"set": "ELEMENT",
                         "type": "MW",
                         "h": 3E-5,
                         "E": 2.07E8,
                         "density": 9.61,
                         "nu": 0.3,
                         "FSI": True}

    elStruc["rib"] = {"set": "ELEMENT",
                      "type": "MW",
                      "h": 3E-5,
                      "E": 2.07E8,
                      "density": 9.61,
                      "nu": 0.3}

    elStruc["sew"] = {"set": "ELEMENT",
                      "type": "CW",
                      "d": 1.3E-4,
                      "E": 2.07E9,
                      "density": 9.61}

    elStruc["bridle"] = {"set": "ELEMENT",
                         "type": "CW",
                         "d": 1.3E-4,
                         "E": 2.07E13,
                         "density": 9.61}

    theta = 90 * np.pi / 180
    g = 9.81

    return {"dim": 3,
            "msh": wing.name,
            "msh_dir": wing.dir,
            "output_dir": output_dir,
            "wrinkling_model": "Jarasjarungkiat",
            "sigma_max": 1E-10,
            "iter_goal": 500,
            "solver": "KDR",
            "maxIterDR": 100,
            "method": "Barnes",
            "follower_pressure": True,
            "lam": 0.9,
            "epsilon_R": 1E-5,
            "epsilon_KE": 1E-10,
            "gravity": False,
            "g": [np.sin(theta) * g, 0, np.cos(theta) * g],
            "elStruc": elStruc}


# --------------------------------------------------------------------------------------------------
# pressure mapping
# --------------------------------------------------------------------------------------------------

def msh_elements(msh_path, gmsh_type, ordered=False):
    """gmsh element number -> sorted (0-based) node tuple, for every element of gmsh_type
    (1 = 2-node line, 2 = 3-node triangle) in the mesh; in the msh node order if ordered."""
    n_nodes = {1: 2, 2: 3}[gmsh_type]
    elements = {}
    with open(msh_path) as f:
        for line in f:
            if line.startswith("$Elements"):
                break
        n = int(next(f))
        for _ in range(n):
            a = next(f).split()
            if a[1] == str(gmsh_type):
                ntags = int(a[2])
                nodes = [int(v) - 1 for v in a[3 + ntags:3 + ntags + n_nodes]]
                elements[int(a[0])] = tuple(nodes if ordered else sorted(nodes))
    return elements


def msh_triangles(msh_path):
    """gmsh element number -> sorted (0-based) node triple, for every triangle in the mesh."""
    return msh_elements(msh_path, 2)


def load_fsi_pressure(csm, wing, internal_pressure=0.0, ratio=1.0):
    """Read the pressure CSV of the wing (triangle = gmsh element number, pressure [Pa]) and return
    the pressure of each FSI triangle of mem4py, in csm.elFSI order:
        p = -ratio * p_csv + internal_pressure
    (the CSV sign is inverted). Triangles are matched by their sorted node triple."""
    csv_path = wing.pressure_csv
    msh_tri = msh_triangles(wing.msh)
    nodes_to_p = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["triangle"])
            if i not in msh_tri:
                raise ValueError("triangle %d of %s is not a triangle in %s" % (i, csv_path, wing.msh))
            nodes_to_p[msh_tri[i]] = float(r["pressure"])

    Nm = np.asarray(csm.Nm)[:csm.nelemsMem]
    elFSI = np.asarray(csm.elFSI)[:csm.nFSI]
    p = np.empty(csm.nFSI)
    missing = 0
    for k, el in enumerate(elFSI):
        key = tuple(sorted(int(n) for n in Nm[el, 1:4]))
        if key in nodes_to_p:
            p[k] = nodes_to_p[key]
        else:
            p[k] = 0.0
            missing += 1
    if missing:
        print("  Warning: %d FSI triangles have no pressure in %s, set to 0 (before P_vent)"
              % (missing, csv_path))
    print("Pressure read from %s: %d triangles" % (csv_path, len(nodes_to_p)))
    return -ratio * p + internal_pressure


# --------------------------------------------------------------------------------------------------
# bridle material
# --------------------------------------------------------------------------------------------------

def apply_bridle_material(csm, wing):
    """If <wing>_bridle_material.csv exists, give every bridle line listed in it (id = gmsh
    geometrical entity) its own diameter [m] and E_secant [Pa], before the solve.

    elStruc["bridle"] holds a single d and E for the whole physical group, so the values are written
    per cable element into the mem4py arrays read by the solver (and by the vtk output):
        area2 = pi * diameter^2 / 4,   E2 = E_secant
    Bridle lines not in the CSV keep d and E of elStruc["bridle"]. The density of elStruc["bridle"]
    is kept, so the line mass follows the new cross-section."""
    path = os.path.join(wing.dir, "%s_bridle_material.csv" % wing.name)
    if not os.path.isfile(path):
        print("No %s, bridle material of elStruc[\"bridle\"] used for every line" % path)
        return
    with open(path, newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    for column in ("id", "diameter", "E_secant"):
        if rows and column not in rows[0]:
            raise ValueError("%s: column %r missing" % (path, column))

    entities = msh_line_entities(wing.msh)
    Nc = np.asarray(csm.Nc)
    set_names = csm.setNames
    cable = {}
    for k in range(csm.nelemsCable):
        cable.setdefault(tuple(sorted((int(Nc[k, 1]), int(Nc[k, 2])))), []).append(k)
    bridle = {k for k in range(csm.nelemsCable) if set_names[Nc[k, 0]] == "bridle"}

    area = np.asarray(csm.area2)                         # shares memory with mem4py: edit in place
    E = np.asarray(csm.E2)
    done, types, skipped = set(), {}, []
    for r in rows:
        line_id = int(r["id"])
        name = r.get("name") or "line %d" % line_id
        if line_id not in entities:
            raise ValueError("%s: no line of geometrical entity %d (%s) in %s" % (path, line_id, name, wing.msh))
        try:
            d, e = float(r["diameter"]), float(r["E_secant"])
        except ValueError:
            d = e = 0.0
        if d <= 0 or e <= 0:        # no material given (e.g. risers: diameter 0, no E_secant)
            skipped.append("%s (%s)" % (name, r.get("linetype") or "-"))
            continue
        ks = [k for pair in entities[line_id] for k in cable.get(pair, [])]
        if len(ks) != len(entities[line_id]):
            raise ValueError("%s: line %d (%s) has segments that are not mem4py cable elements" % (path, line_id, name))
        not_bridle = [k for k in ks if k not in bridle]
        if not_bridle:
            raise ValueError("%s: line %d (%s) is not in the bridle physical group" % (path, line_id, name))
        area[ks] = np.pi * d ** 2 / 4
        E[ks] = e
        done.update(ks)
        t = types.setdefault(r.get("linetype") or "-", [d, e, 0, 0])
        t[2] += 1
        t[3] += len(ks)

    print("Bridle material from %s: %d lines, %d of %d bridle segments"
          % (path, len(rows) - len(skipped), len(done), len(bridle)))
    if skipped:
        print("  %d lines without diameter / E_secant keep elStruc[\"bridle\"]: %s"
              % (len(skipped), ", ".join(skipped)))
    for linetype, (d, e, n_lines, n_seg) in sorted(types.items()):
        print("  %-20s d %.4g mm, E %.4g GPa, EA %.4g N: %d lines, %d segments"
              % (linetype, 1E3 * d, 1E-9 * e, e * np.pi * d ** 2 / 4, n_lines, n_seg))
    if len(done) < len(bridle):
        print("  %d bridle segments without a material in the CSV keep elStruc[\"bridle\"] (d %g m, E %g Pa)"
              % (len(bridle) - len(done), csm.elStruc["bridle"]["d"], csm.elStruc["bridle"]["E"]))


# --------------------------------------------------------------------------------------------------
# line drag
# --------------------------------------------------------------------------------------------------

def free_stream(it):
    """Free stream velocity [m/s] of a json iteration in the structure mesh frame.

    The CFD rotates the wing by R = Rz(Wing_Yaw) Ry(Wing_Pitch) Rx(Wing_Roll) (wing_fsi_cfd.rotation)
    and blows Speed * (cos Glide_angle, 0, sin Glide_angle) onto it (wing_openfoam.py), so in the
    unrotated mesh of mem4py the flow is R^T times that vector."""
    r, p, y = np.radians([float(it.get(k, 0.0)) for k in ("Wing_Roll", "Wing_Pitch", "Wing_Yaw")])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    a = np.radians(float(it["Glide_angle"]))
    return (Rz @ Ry @ Rx).T @ (float(it["Speed"]) * np.array([np.cos(a), 0.0, np.sin(a)]))


def apply_line_drag(csm, U, Cd=LINE_CD, rho=RHO_AIR, sets=LINE_DRAG_SETS):
    """Cross-flow drag of the cable elements of the given physical groups, added to the external
    nodal force vector of mem4py (csm.force_vector, added to the calculated forces at every
    assembly), half of each segment force on each of its nodes:
        F = 1/2 rho Cd d L |U_n| U_n,   U_n = U - (U.t) t   (flow normal to the segment, t = unit tangent)
    d is the diameter of each element (from csm.area2, so call after apply_bridle_material) and L its
    undeformed length. The force is computed once on the undeformed lines and keeps its value during
    the solve (not a follower load); it is only exact with nLoadSteps = 1 (mem4py scales
    force_vector in place at every load step)."""
    U = np.asarray(U, dtype=float)
    Nc = np.asarray(csm.Nc)[:csm.nelemsCable]
    P = np.column_stack((csm.X0, csm.Y0, csm.Z0))
    d = np.sqrt(4 * np.asarray(csm.area2) / np.pi)
    f = np.asarray(csm.force_vector)                     # shares memory with mem4py: edit in place

    names = csm.setNames
    ks = [k for k in range(csm.nelemsCable) if names[Nc[k, 0]] in sets]
    if not ks:
        print("Line drag: no cable element in %s, nothing applied" % ", ".join(sets))
        return
    a, b = Nc[ks, 1].astype(int), Nc[ks, 2].astype(int)
    t = P[b] - P[a]
    L = np.linalg.norm(t, axis=1)
    t /= L[:, None]
    Un = U - (t @ U)[:, None] * t
    F = 0.5 * rho * Cd * (d[ks] * L * np.linalg.norm(Un, axis=1))[:, None] * Un
    for c in range(3):
        np.add.at(f, 3 * a + c, F[:, c] / 2)
        np.add.at(f, 3 * b + c, F[:, c] / 2)

    total = F.sum(axis=0)
    print("Line drag (%s, Cd %g, rho %g kg/m3): U = (%.3g, %.3g, %.3g) m/s, %d segments, %.4g m of line"
          % (", ".join(sets), Cd, rho, U[0], U[1], U[2], len(ks), L.sum()))
    print("  total force (%.2f, %.2f, %.2f) N, |F| %.2f N, %.2f N along the flow"
          % (total[0], total[1], total[2], np.linalg.norm(total), total @ U / np.linalg.norm(U)))


# --------------------------------------------------------------------------------------------------
# line pre-tension
# --------------------------------------------------------------------------------------------------

def msh_line_entities(msh_path):
    """gmsh geometrical entity (second tag) -> sorted (0-based) node pairs of its 2-node lines."""
    entities = {}
    with open(msh_path) as f:
        for line in f:
            if line.startswith("$Elements"):
                break
        n = int(next(f))
        for _ in range(n):
            a = next(f).split()
            if a[1] == "1":
                ntags = int(a[2])
                nodes = tuple(sorted(int(v) - 1 for v in a[3 + ntags:5 + ntags]))
                entities.setdefault(int(a[4]), []).append(nodes)
    return entities


def apply_pretension(csm, wing, csv_path):
    """Shorten the rest length of the lines of csv_path (columns id = gmsh geometrical entity of the
    line, stretch [%], optional name and length) to L_mesh * (1 + stretch / 100), before the solve:
    stretch -50 makes the line 50 % shorter (pre-tensioned), +10 makes it 10 % longer (slack). The
    change is applied to every segment of the line, so the whole line changes uniformly."""
    with open(csv_path, newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    for column in ("id", "stretch"):
        if rows and column not in rows[0]:
            raise ValueError("%s: column %r missing" % (csv_path, column))

    entities = msh_line_entities(wing.msh)
    Nc = np.asarray(csm.Nc)
    cable = {}
    for k in range(csm.nelemsCable):
        cable.setdefault(tuple(sorted((int(Nc[k, 1]), int(Nc[k, 2])))), []).append(k)

    L0 = np.asarray(csm.LCable)                          # shares memory with mem4py: edit in place
    print("Line pre-tension from %s:" % csv_path)
    for r in rows:
        line_id, stretch = int(r["id"]), float(r["stretch"])
        name = r.get("name") or "line %d" % line_id
        if line_id not in entities:
            raise ValueError("%s: no line of geometrical entity %d (%s) in %s" % (csv_path, line_id, name, wing.msh))
        if stretch <= -100:
            raise ValueError("%s: stretch of line %d (%s) must be > -100 %%" % (csv_path, line_id, name))
        ks = [k for pair in entities[line_id] for k in cable.get(pair, [])]
        if len(ks) != len(entities[line_id]):
            raise ValueError("%s: line %d (%s) has segments that are not mem4py cable elements"
                             % (csv_path, line_id, name))
        L_mesh = L0[ks].sum()
        if r.get("length") and abs(L_mesh - float(r["length"])) > 1E-3 * L_mesh:
            print("  Warning: %s (line %d) is %.6g m long in the mesh, %s m in the CSV"
                  % (name, line_id, L_mesh, r["length"]))
        L0[ks] *= 1.0 + stretch / 100.0
        print("  %-14s line %4d (%d segments): mesh %.6g m, rest length %.6g m (%+g %%)"
              % (name, line_id, len(ks), L_mesh, L0[ks].sum(), stretch))


# --------------------------------------------------------------------------------------------------
# picture
# --------------------------------------------------------------------------------------------------

def plot_pressure(csm, p, path, title):
    """3D views of the wing, canopy coloured by pressure, ribs in grey, lines in black."""
    P = np.column_stack((np.asarray(csm.X), np.asarray(csm.Y), np.asarray(csm.Z)))
    Nm = np.asarray(csm.Nm)[:csm.nelemsMem]
    elFSI = np.asarray(csm.elFSI)[:csm.nFSI]
    canopy = P[Nm[elFSI, 1:4]]

    is_fsi = np.zeros(len(Nm), dtype=bool)
    is_fsi[elFSI] = True
    ribs = P[Nm[~is_fsi, 1:4]]
    lines = P[np.asarray(csm.Nc)[:csm.nelemsCable, 1:3]] if csm.nelemsCable > 0 else np.empty((0, 2, 3))

    # diverging colour map centred on zero: suction (p < 0) blue, overpressure (p > 0) red
    norm = TwoSlopeNorm(vmin=min(p.min(), -1E-12), vcenter=0.0, vmax=max(p.max(), 1E-12))
    cmap = matplotlib.colormaps["RdBu_r"]
    colors = cmap(norm(p))

    lo, hi = P.min(axis=0), P.max(axis=0)
    views = [("perspective", 25, -60), ("top surface", 89, -90), ("bottom surface", -89, -90),
             ("front", 0, 0)]

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    for i, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(ribs, facecolors=(0.75, 0.75, 0.75, 0.25),
                                             edgecolors="none"))
        ax.add_collection3d(Poly3DCollection(canopy, facecolors=colors, edgecolors="none"))
        if len(lines) and name != "bottom surface":     # lines would hide the bottom surface
            ax.add_collection3d(Line3DCollection(lines, colors="k", linewidths=0.3, alpha=0.5))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.6, location="right")
    cb.set_label("FSI pressure [Pa]")
    fig.suptitle("%s   (min %.4g, max %.4g)" % (title, p.min(), p.max()), fontsize=14)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("picture written to %s" % path)


# --------------------------------------------------------------------------------------------------
# vtk export per part
# --------------------------------------------------------------------------------------------------

def _parse_floats(lines):
    return np.array(" ".join(lines).split(), dtype=float)


def read_mem4py_vtk(path):
    """Read the legacy ASCII vtk written by mem4py.

    Returns a dict with "points" (n, 3) undeformed coordinates, "cells" (list of node index lists),
    "cell_types" (m,), "point_data" and "cell_data" (name -> (kind, vtk type, array)), kind being
    "VECTORS" (array (n, 3)) or "SCALARS" (array (n,))."""
    with open(path) as f:
        lines = f.read().split("\n")

    vtk = {"point_data": {}, "cell_data": {}}
    section, count, i = None, 0, 0
    while i < len(lines):
        tokens = lines[i].split()
        if not tokens:
            i += 1
            continue
        key = tokens[0]
        if key == "POINTS":
            n = int(tokens[1])
            vtk["points"] = _parse_floats(lines[i + 1:i + 1 + n]).reshape(n, 3)
            i += 1 + n
        elif key == "CELLS":
            n = int(tokens[1])
            vtk["cells"] = [[int(v) for v in l.split()[1:]] for l in lines[i + 1:i + 1 + n]]
            i += 1 + n
        elif key == "CELL_TYPES":
            n = int(tokens[1])
            vtk["cell_types"] = np.array(lines[i + 1:i + 1 + n], dtype=int)
            i += 1 + n
        elif key in ("POINT_DATA", "CELL_DATA"):
            section, count = ("point_data" if key == "POINT_DATA" else "cell_data"), int(tokens[1])
            i += 1
        elif key == "VECTORS":
            vtk[section][tokens[1]] = ("VECTORS", tokens[2],
                                       _parse_floats(lines[i + 1:i + 1 + count]).reshape(count, 3))
            i += 1 + count
        elif key == "SCALARS":                  # followed by a LOOKUP_TABLE line
            vtk[section][tokens[1]] = ("SCALARS", tokens[2], _parse_floats(lines[i + 2:i + 2 + count]))
            i += 2 + count
        else:
            i += 1
    return vtk


# element kind of a part CSV, given by the header of its first column:
#   header -> (gmsh element type, vtk cell type, nodes per cell, mem4py cell fields NOT exported)
ELEMENT_KINDS = {"triangle": (2, 5, 3, "Cable_"),
                 "line": (1, 3, 2, "Membrane_")}


def read_part_csv(csv_path):
    """(kind, gmsh element numbers) of a part CSV whose first column is "triangle" or "line"."""
    with open(csv_path, newline="") as f:
        rows = csv.reader(f)
        kind = next(rows)[0].strip()
        if kind not in ELEMENT_KINDS:
            raise ValueError("%s: first column must be one of %s, not %r"
                             % (csv_path, ", ".join(ELEMENT_KINDS), kind))
        return kind, np.array([int(r[0]) for r in rows if r], dtype=int)


def read_part_triangles(csv_path):
    """gmsh element numbers of the first column of a part CSV (triangle,piece)."""
    return read_part_csv(csv_path)[1]


def _write_field(fout, name, kind, vtk_type, values):
    if kind == "VECTORS":
        fout.write("VECTORS %s %s\n" % (name, vtk_type))
        np.savetxt(fout, values, fmt="%.10g")
    else:
        fout.write("SCALARS %s %s 1\nLOOKUP_TABLE default\n" % (name, vtk_type))
        np.savetxt(fout, values, fmt="%d" if vtk_type == "int" else "%.10g")
    fout.write("\n")


def write_part_vtk(path, vtk, cell_index, elem_id, element_kind, title):
    """Write the cells cell_index of vtk (triangles or lines, see ELEMENT_KINDS), deformed by the
    point field u, keeping only the used points, with the gmsh element numbers elem_id as cell
    field named after element_kind ("triangle" or "line")."""
    _, vtk_cell_type, n_nodes, skip_prefix = ELEMENT_KINDS[element_kind]
    conn = np.array([vtk["cells"][c] for c in cell_index], dtype=int)
    used, local = np.unique(conn, return_inverse=True)
    local = local.reshape(-1, n_nodes)
    points = vtk["points"][used] + vtk["point_data"]["u"][2][used]

    with open(path, "w") as fout:
        fout.write("# vtk DataFile Version 2.0\n%s\nASCII\nDATASET UNSTRUCTURED_GRID\n" % title)
        fout.write("POINTS %d double\n" % len(points))
        np.savetxt(fout, points, fmt="%.10g")
        fout.write("\nCELLS %d %d\n" % (len(local), (n_nodes + 1) * len(local)))
        np.savetxt(fout, np.column_stack((np.full(len(local), n_nodes), local)), fmt="%d")
        fout.write("\nCELL_TYPES %d\n" % len(local))
        np.savetxt(fout, np.full(len(local), vtk_cell_type), fmt="%d")     # 5 = triangle, 3 = line

        fout.write("\nPOINT_DATA %d\n" % len(points))
        for name, (kind, vtk_type, values) in vtk["point_data"].items():
            _write_field(fout, name, kind, vtk_type, values[used])

        fout.write("CELL_DATA %d\n" % len(local))
        _write_field(fout, element_kind, "SCALARS", "int", elem_id)
        for name, (kind, vtk_type, values) in vtk["cell_data"].items():
            if not name.startswith(skip_prefix):    # cable fields are meaningless on triangles and vice versa
                _write_field(fout, name, kind, vtk_type, values[cell_index])


def latest_vtk(wing, out_dir):
    """Last mem4py vtk result <wing>_NNNNN.vtk of out_dir."""
    files = sorted(glob.glob(os.path.join(out_dir, wing.name + "_[0-9]*.vtk")))
    if not files:
        raise FileNotFoundError("no %s_NNNNN.vtk result in %s" % (wing.name, out_dir))
    return files[-1]


def write_physical_csv(wing, name):
    """Write <wing>_<name>.csv listing the elements of the gmsh physical group name, as
    "line,piece" (1D group) or "triangle,piece" (2D group), piece = "<name>:<gmsh entity tag>"."""
    path = os.path.join(wing.dir, "%s_%s.csv" % (wing.name, name))
    with open(wing.msh) as f:
        for line in f:
            if line.startswith("$PhysicalNames"):
                break
        else:
            raise ValueError("%s has no $PhysicalNames" % wing.msh)
        groups = {}
        for _ in range(int(next(f))):
            dim, tag, pname = next(f).split(maxsplit=2)
            groups[pname.strip().strip('"')] = (int(dim), tag)
        if name not in groups:
            raise ValueError("no physical group %r in %s" % (name, wing.msh))
        dim, tag = groups[name]
        kind, gmsh_type = {1: ("line", "1"), 2: ("triangle", "2")}[dim]

        for line in f:
            if line.startswith("$Elements"):
                break
        rows = []
        for _ in range(int(next(f))):
            a = next(f).split()
            if a[1] == gmsh_type and int(a[2]) >= 2 and a[3] == tag:     # tags: physical, entity
                rows.append((a[0], "%s:%s" % (name, a[4])))

    with open(path, "w", newline="") as f:
        out = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        f.write("%s,piece\n" % kind)
        for elem, piece in rows:
            out.writerow((int(elem), piece))
    print("  %s written: %d %ss of physical group %s" % (path, len(rows), kind, name))


def export_parts(wing, vtk_path, out_dir):
    """Split the mem4py vtk result into <wing>_<part>.vtk, one per <wing>_<part>.csv."""
    print("Splitting %s per part" % vtk_path)
    vtk = read_mem4py_vtk(vtk_path)

    for name in PHYSICAL_PARTS:
        if not os.path.isfile(os.path.join(wing.dir, "%s_%s.csv" % (wing.name, name))):
            write_physical_csv(wing, name)

    nodes_to_cell, msh_elem = {}, {}       # per element kind, built on first use
    for part in PARTS:
        csv_path = os.path.join(wing.dir, "%s_%s.csv" % (wing.name, part))
        if not os.path.isfile(csv_path):
            print("  %s not found, skipped" % csv_path)
            continue
        kind, elem_id = read_part_csv(csv_path)
        gmsh_type, vtk_cell_type = ELEMENT_KINDS[kind][:2]

        if kind not in nodes_to_cell:
            # mem4py cells of this kind, identified by their sorted node tuple
            cells = {}
            for c in np.flatnonzero(vtk["cell_types"] == vtk_cell_type):
                key = tuple(sorted(vtk["cells"][c]))
                cells.setdefault(key, []).append(c)
            duplicated = sum(1 for c in cells.values() if len(c) > 1)
            if duplicated:
                print("  Warning: %d %ss of the vtk result share their nodes with another one, "
                      "their fields may be mixed up" % (duplicated, kind))
            nodes_to_cell[kind] = {k: c[0] for k, c in cells.items()}
            msh_elem[kind] = msh_elements(wing.msh, gmsh_type)

        to_cell, msh_nodes = nodes_to_cell[kind], msh_elem[kind]
        unknown = [i for i in elem_id if i not in msh_nodes or msh_nodes[i] not in to_cell]
        if unknown:
            raise ValueError("%d %ss of %s are not in the vtk result (first: %d)"
                             % (len(unknown), kind, csv_path, unknown[0]))
        cell_index = np.array([to_cell[msh_nodes[i]] for i in elem_id], dtype=int)

        path = os.path.join(out_dir, "%s_%s.vtk" % (wing.name, part))
        write_part_vtk(path, vtk, cell_index, elem_id, kind, "%s %s (deformed)" % (wing.name, part))
        print("  %s: %d %ss written to %s" % (part, len(elem_id), kind, path))


def read_csv_pressure(wing):
    """gmsh triangle number -> pressure [Pa] of the pressure CSV (as written, before P_ratio)."""
    with open(wing.pressure_csv, newline="") as f:
        return {int(r["triangle"]): float(r["pressure"]) for r in csv.DictReader(f)}


def write_force_vtk(wing, vtk_path, out_dir, ratio, internal_pressure):
    """Write <wing>_force.vtk: one line per triangle of <wing>_canopy.csv showing the pressure force
    on that triangle at the deformed position (undeformed points + u of the mem4py vtk result).

    The pressure is the one applied in mem4py, p = -ratio * p_csv + internal_pressure (internal
    pressure only for triangles missing from the pressure CSV). As in mem4py, the force is
        F = p * A * n,   n = (x2 - x1) x (x3 - x1) / |(x2 - x1) x (x3 - x1)|  (msh node order)
    and the line goes from the triangle centre to centre + F, so it is |F| long (1 N = 1 m) and
    lies along the normal (along -n when p < 0). Cell fields: force = p * A [N] (signed, its
    absolute value is the line length), F [N] force vector, pressure [Pa], area [m2], triangle
    (gmsh element number)."""
    canopy_csv = os.path.join(wing.dir, "%s_canopy.csv" % wing.name)
    kind, tri_id = read_part_csv(canopy_csv)
    if kind != "triangle":
        raise ValueError("%s must list triangles to write the force vtk" % canopy_csv)

    msh_tri = msh_elements(wing.msh, 2, ordered=True)
    unknown = [i for i in tri_id if i not in msh_tri]
    if unknown:
        raise ValueError("%d triangles of %s are not in %s (first: %d)"
                         % (len(unknown), canopy_csv, wing.msh, unknown[0]))
    p_csv = read_csv_pressure(wing)
    missing = sum(1 for i in tri_id if i not in p_csv)
    if missing:
        print("  Warning: %d canopy triangles have no pressure in %s, only P_vent applied"
              % (missing, wing.pressure_csv))
    p = np.array([-ratio * p_csv.get(i, 0.0) + internal_pressure for i in tri_id])

    vtk = read_mem4py_vtk(vtk_path)                     # point index = msh node - 1
    X = vtk["points"] + vtk["point_data"]["u"][2]       # deformed
    tri = X[np.array([msh_tri[i] for i in tri_id], dtype=int)]   # (n, 3 nodes, xyz)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    F = 0.5 * p[:, None] * cross                         # = p * A * n
    centre = tri.mean(axis=1)

    n = len(tri_id)
    points = np.empty((2 * n, 3))
    points[0::2], points[1::2] = centre, centre + F
    path = os.path.join(out_dir, "%s_force.vtk" % wing.name)
    with open(path, "w") as fout:
        fout.write("# vtk DataFile Version 2.0\n%s canopy pressure force (deformed)\nASCII\n"
                   "DATASET UNSTRUCTURED_GRID\n" % wing.name)
        fout.write("POINTS %d double\n" % len(points))
        np.savetxt(fout, points, fmt="%.10g")
        fout.write("\nCELLS %d %d\n" % (n, 3 * n))
        np.savetxt(fout, np.column_stack((np.full(n, 2), np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2))), fmt="%d")
        fout.write("\nCELL_TYPES %d\n" % n)
        np.savetxt(fout, np.full(n, 3), fmt="%d")       # 3 = line
        fout.write("\nCELL_DATA %d\n" % n)
        _write_field(fout, "force", "SCALARS", "double", p * area)
        _write_field(fout, "F", "VECTORS", "double", F)
        _write_field(fout, "pressure", "SCALARS", "double", p)
        _write_field(fout, "area", "SCALARS", "double", area)
        _write_field(fout, "triangle", "SCALARS", "int", tri_id)
    print("  force: %d canopy triangles written to %s (total |F| %.4g N, resultant %s N)"
          % (n, path, np.linalg.norm(F, axis=1).sum(), np.array2string(F.sum(axis=0), precision=4)))
    return centre, F, area


def write_force_sum_vtk(wing, out_dir, centre, F, area, scale=FORCE_SUM_SCALE):
    """Write <wing>_force_sum.vtk: one line showing the resultant of the triangle forces of
    <wing>_force.vtk (centre c_i, force F_i, area A_i as returned by write_force_vtk).

        R   = sum F_i                         resultant force [N]
        M_O = sum c_i x F_i                   moment about the origin [N m]
        P0  = (R x M_O) / |R|^2               point of the line of action (central axis) closest
                                              to the origin; the line is P0 + t R
        P   = point of that line closest to the canopy area centroid
        M_par = (M_O . R) / |R|               remaining twist about the line [N m]

    The line goes from P to P + scale * R (scale [m/N], FORCE_SUM_SCALE = 0.005 by default, so
    1000 N is 5 m long). Cell fields: force = |R| [N], F = R [N], moment = M_O [N m],
    twist = M_par [N m], triangles = number of summed triangles."""
    R = F.sum(axis=0)
    M = np.cross(centre, F).sum(axis=0)
    R2 = R.dot(R)
    if R2 == 0.0:
        raise ValueError("resultant canopy force is zero, %s_force_sum.vtk not written" % wing.name)
    u = R / np.sqrt(R2)
    P0 = np.cross(R, M) / R2
    g = (centre * area[:, None]).sum(axis=0) / area.sum()
    P = P0 + u * np.dot(g - P0, u)
    twist = M.dot(u)

    path = os.path.join(out_dir, "%s_force_sum.vtk" % wing.name)
    with open(path, "w") as fout:
        fout.write("# vtk DataFile Version 2.0\n%s canopy resultant force (deformed, %g m/N)\nASCII\n"
                   "DATASET UNSTRUCTURED_GRID\n" % (wing.name, scale))
        fout.write("POINTS 2 double\n")
        np.savetxt(fout, np.array([P, P + scale * R]), fmt="%.10g")
        fout.write("\nCELLS 1 3\n2 0 1\n\nCELL_TYPES 1\n3\n\nCELL_DATA 1\n")
        _write_field(fout, "force", "SCALARS", "double", np.array([np.sqrt(R2)]))
        _write_field(fout, "F", "VECTORS", "double", R[None, :])
        _write_field(fout, "moment", "VECTORS", "double", M[None, :])
        _write_field(fout, "twist", "SCALARS", "double", np.array([twist]))
        _write_field(fout, "triangles", "SCALARS", "int", np.array([len(F)]))
    print("  force sum written to %s (%g m/N):" % (path, scale))
    print("    R = (%.4g, %.4g, %.4g) N, |R| = %.4g N" % (R[0], R[1], R[2], np.sqrt(R2)))
    print("    M_O = (%.4g, %.4g, %.4g) N m, line of action through P = (%.4g, %.4g, %.4g) m"
          % (M[0], M[1], M[2], P[0], P[1], P[2]))
    print("    twist about the line %.4g N m (%.4g m x |R|)" % (twist, twist / np.sqrt(R2)))
    return R


def write_canopy_forces(wing, vtk_path, out_dir, ratio, internal_pressure):
    """<wing>_force.vtk and <wing>_force_sum.vtk, plus an equilibrium check against the reactions
    at main_l / main_r of <wing>_attachments.csv when that file exists."""
    R = write_force_sum_vtk(wing, out_dir, *write_force_vtk(wing, vtk_path, out_dir, ratio, internal_pressure))
    if os.path.isfile(os.path.join(wing.dir, "%s_attachments.csv" % wing.name)):
        att = read_attachments(wing)
        reaction = read_mem4py_vtk(vtk_path)["point_data"]["F_reaction"][2]
        mains = sum(reaction[att[n]] for n in ("main_l", "main_r") if n in att)
        print("    main_l + main_r reaction = (%.4g, %.4g, %.4g) N, |.| = %.4g N (vs |R| %.4g N)"
              % (mains[0], mains[1], mains[2], np.linalg.norm(mains), np.linalg.norm(R)))


# --------------------------------------------------------------------------------------------------
# next iteration of the json
# --------------------------------------------------------------------------------------------------

def read_main_risers(wing):
    """{riser name: (main node, upper node)} from the main_riser rows of <wing>_data.csv, nodes as
    0-based mem4py indices (msh node - 1)."""
    path = os.path.join(wing.dir, "%s_data.csv" % wing.name)
    risers = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].startswith("main_riser "):
                continue
            _, name, key = row[0].split()
            if key in ("main_node", "upper_node"):
                risers.setdefault(name, {})[key] = int(row[1]) - 1
    if not risers:
        raise ValueError("no main_riser rows in %s" % path)
    for name, nodes in risers.items():
        if len(nodes) != 2:
            raise ValueError("main_riser %s of %s needs a main_node and an upper_node" % (name, path))
    return {name: (n["main_node"], n["upper_node"]) for name, n in risers.items()}


def read_attachments(wing):
    """{name: 0-based mem4py node index (msh node - 1)} from <wing>_attachments.csv."""
    path = os.path.join(wing.dir, "%s_attachments.csv" % wing.name)
    with open(path, newline="") as f:
        return {r["name"].strip(): int(r["node"]) - 1 for r in csv.DictReader(f)}


def write_attachment_forces(wing, vtk_path, out_dir):
    """Write <wing>_attachments_force.csv in out_dir: one row per node of <wing>_attachments.csv
    (same order, node / name / type copied) with

        lines           number of bridle segments (of <wing>_bridle.csv) ending at the node
        Fx, Fy, Fz, F   pull of those bridle segments on the node [N]:
                        sum of Cable_force * (x_other - x_node) / |x_other - x_node| at the deformed
                        position, i.e. pointing along the lines away from the node
        Rx, Ry, Rz, R   reaction force F_reaction of mem4py at the node [N] (non-zero only for
                        fixed nodes such as main_l / main_r)

    Only the bridle is counted, so sew or strap cables that also end at a canopy node are left out."""
    path_in = os.path.join(wing.dir, "%s_attachments.csv" % wing.name)
    if not os.path.isfile(path_in):
        print("  %s not found, attachment forces skipped" % path_in)
        return
    with open(path_in, newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]

    bridle_csv = os.path.join(wing.dir, "%s_bridle.csv" % wing.name)
    kind, line_id = read_part_csv(bridle_csv)
    if kind != "line":
        raise ValueError("%s must list lines to compute the attachment forces" % bridle_csv)
    bridle = {tuple(sorted(n)) for n in (msh_elements(wing.msh, 1)[i] for i in line_id)}

    vtk = read_mem4py_vtk(vtk_path)                     # point index = msh node - 1
    X = vtk["points"] + vtk["point_data"]["u"][2]       # deformed
    reaction = vtk["point_data"]["F_reaction"][2]
    tension = vtk["cell_data"]["Cable_force"][2]
    pull = np.zeros_like(X)
    count = np.zeros(len(X), dtype=int)
    for c in np.flatnonzero(vtk["cell_types"] == 3):
        a, b = vtk["cells"][c]
        if tuple(sorted((a, b))) not in bridle:
            continue
        d = X[b] - X[a]
        f = tension[c] * d / np.linalg.norm(d)
        pull[a] += f
        pull[b] -= f
        count[a] += 1
        count[b] += 1

    path = os.path.join(out_dir, "%s_attachments_force.csv" % wing.name)
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["node", "name", "type", "lines", "Fx", "Fy", "Fz", "F", "Rx", "Ry", "Rz", "R"])
        for r in rows:
            n = int(r["node"]) - 1
            out.writerow([r["node"], r["name"], r.get("type", ""), count[n]]
                         + ["%.6g" % v for v in pull[n]] + ["%.6g" % np.linalg.norm(pull[n])]
                         + ["%.6g" % v for v in reaction[n]] + ["%.6g" % np.linalg.norm(reaction[n])])
    unlinked = [r["name"] for r in rows if count[int(r["node"]) - 1] == 0]
    print("  attachment forces: %d nodes written to %s" % (len(rows), path))
    if unlinked:
        print("  Warning: no bridle line at %s" % ", ".join(unlinked))


def main_riser(wing, out_dir):
    """Main attachment forces and rotation of the main risers.

    The force of each side is the magnitude of the reaction force F_reaction at the node main_l
    (left) or main_r (right) of <wing>_attachments.csv.

    Each main riser of <wing>_data.csv goes from its main_node (pilot attachment point) to its
    upper_node. Its rotation is the change, between the undeformed and the deformed position, of
    the riser angle in the side (x-z) plane, atan2(dx, dz): positive when the upper node moves
    toward +x relative to the main node. The returned rotation is the mean over all main risers.

    Returns ({"left": force [N], "right": force [N]}, rotation [deg], {riser name: rotation [deg]})."""
    vtk = read_mem4py_vtk(latest_vtk(wing, out_dir))   # all points, index = msh node - 1

    attachments = read_attachments(wing)
    reaction = vtk["point_data"]["F_reaction"][2]
    forces = {}
    for side, name in (("left", "main_l"), ("right", "main_r")):
        if name not in attachments:
            raise ValueError("no node %s in %s_attachments.csv" % (name, wing.name))
        forces[side] = float(np.linalg.norm(reaction[attachments[name]]))

    X = vtk["points"] + vtk["point_data"]["u"][2]       # deformed (vtk points are undeformed)
    X0 = vtk["points"]
    rotations = {}
    for name, (start, end) in read_main_risers(wing).items():
        d0, d = X0[end] - X0[start], X[end] - X[start]
        r = np.degrees(np.arctan2(d[0], d[2]) - np.arctan2(d0[0], d0[2]))
        rotations[name] = float((r + 180.0) % 360.0 - 180.0)   # atan2 jumps by 360 deg around +-180 deg
    return forces, float(np.mean(list(rotations.values()))), rotations


def prepare_next_iteration(wing, iteration, out_dir):
    """Add (or replace) iteration + 1 in the json, from the result of iteration:
        P_ratio     = NEXT_P_RATIO
        P_vent      = NEXT_P_VENT
        Force_main_left / Force_main_right = reaction force at main_l / main_r of <wing>_attachments.csv
        Wing_Pitch  = mean rotation of the main risers of <wing>_data.csv during this iteration [deg]
    the other keys are copied unchanged."""
    with open(wing.json) as f:
        doc = json.load(f)
    current = next(it for it in doc["iterations"] if it["iteration"] == iteration)

    forces, rotation, rotations = main_riser(wing, out_dir)

    nxt = dict(current)
    nxt["iteration"] = iteration + 1
    nxt.pop("Wing_angle", None)        # old json keys, replaced by Wing_Pitch and Force_main_left/right
    nxt.pop("Force_main", None)
    nxt["Wing_Pitch"] = rotation
    nxt["Force_main_right"] = forces["right"]
    nxt["Force_main_left"] = forces["left"]
    nxt["P_ratio"] = NEXT_P_RATIO
    nxt["P_vent"] = NEXT_P_VENT

    doc["iterations"] = [it for it in doc["iterations"] if it["iteration"] != iteration + 1] + [nxt]
    doc["iterations"].sort(key=lambda it: it["iteration"])
    with open(wing.json, "w") as f:
        json.dump(doc, f, indent=4)
        f.write("\n")

    print("Next iteration %d written to %s:" % (iteration + 1, wing.json))
    for name, r in rotations.items():
        print("  main riser %s rotation %.4g deg" % (name, r))
    print("  P_ratio %.4g, P_vent %.4g Pa, Force_main_left %.4g N, Force_main_right %.4g N, Wing_Pitch %.4g deg"
          % (nxt["P_ratio"], nxt["P_vent"], nxt["Force_main_left"], nxt["Force_main_right"], nxt["Wing_Pitch"]))


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solve", action="store_true",
                        help="solve the structure with the FSI pressure and plot the deformed wing")
    parser.add_argument("--wing", default=WING_DIR,
                        help="wing directory containing <wing>.msh, <wing>_pressure.csv and <wing>.json "
                             "(default %s)" % WING_DIR)
    parser.add_argument("--ratio", type=float, default=None,
                        help="multiplier applied to the CSV pressure (default: P_ratio of the json)")
    parser.add_argument("--iteration", type=int, default=JSON_ITERATION,
                        help="json iteration giving P_ratio and P_vent, output in <wing>/iter_NN/ "
                             "(default %d)" % JSON_ITERATION)
    parser.add_argument("--pretension", nargs="?", const="", default=None, metavar="CSV",
                        help="pre-tension the lines given in CSV (id = gmsh entity, stretch = length change in %%, -50 = 50%% shorter), default CSV "
                             "<wing>/<wing>_control.csv")
    parser.add_argument("--export", action="store_true",
                        help="do not solve, only split the existing vtk result of <wing>/iter_NN/ per part")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start = time.time()

    wing = Wing(args.wing)
    ratio, internal_pressure = wing.iteration_parameters(args.iteration)
    print("%s, iteration %d: P_ratio %g, P_vent %g Pa" % (wing.json, args.iteration, ratio, internal_pressure))
    if args.ratio is not None:
        ratio = args.ratio
        print("P_ratio overridden by --ratio: %g" % ratio)

    out_dir = os.path.join(wing.dir, "iter_%02d" % args.iteration)
    print("Output directory: %s" % out_dir)

    if args.export:
        export_parts(wing, latest_vtk(wing, out_dir), out_dir)
        write_canopy_forces(wing, latest_vtk(wing, out_dir), out_dir, ratio, internal_pressure)
        write_attachment_forces(wing, latest_vtk(wing, out_dir), out_dir)
        prepare_next_iteration(wing, args.iteration, out_dir)
        print("Finished in %.1f s" % (time.time() - start))
        return

    csm = m4p.Mem4py(paraglider_problem(wing, out_dir))
    print("Canopy: %d FSI triangles" % csm.nFSI)

    apply_bridle_material(csm, wing)
    apply_line_drag(csm, free_stream(wing.iteration(args.iteration)))

    if args.pretension is not None:
        apply_pretension(csm, wing, args.pretension or os.path.join(wing.dir, wing.name + "_control.csv"))

    pFSI = load_fsi_pressure(csm, wing, internal_pressure, ratio)
    print("CSV pressure multiplied by %g" % ratio)
    print("Internal pressure %g Pa added to every canopy triangle" % internal_pressure)
    np.asarray(csm.pFSI)[:csm.nFSI] = pFSI       # write into mem4py's own pressure buffer
    print("FSI pressure: min %.4g, mean %.4g, max %.4g Pa" % (pFSI.min(), pFSI.mean(), pFSI.max()))

    plot_pressure(csm, pFSI, os.path.join(out_dir, "paraglider_pressure.png"),
                  "Paraglider FSI pressure (undeformed)")

    if args.solve:
        def plot_peak(model, n_peak):
            # a picture that cannot be written (e.g. file locked by a viewer) must not stop the solver
            path = os.path.join(out_dir, "peaks", "paraglider_peak_%03d.png" % n_peak)
            for attempt in range(2):
                try:
                    plot_pressure(model, pFSI, path, "Paraglider FSI pressure (kinetic energy peak %d)" % n_peak)
                    return
                except OSError as e:
                    plt.close("all")
                    if attempt == 0:
                        time.sleep(1)
                    else:
                        print("Warning: picture of peak %d not written (%s)" % (n_peak, e))

        csm.peak_callback = plot_peak       # picture of the deformed wing at every kinetic energy peak
        csm.solve()
        plot_pressure(csm, pFSI, os.path.join(out_dir, "paraglider_pressure_deformed.png"),
                      "Paraglider FSI pressure (deformed)")
        export_parts(wing, latest_vtk(wing, out_dir), out_dir)
        write_canopy_forces(wing, latest_vtk(wing, out_dir), out_dir, ratio, internal_pressure)
        write_attachment_forces(wing, latest_vtk(wing, out_dir), out_dir)
        prepare_next_iteration(wing, args.iteration, out_dir)

    print("Finished in %.1f s" % (time.time() - start))


if __name__ == "__main__":
    main()
