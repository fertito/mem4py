"""
Single-skin wings: ram air / single skin zones of the canopy.

The wing type is the wing_type row of <wing>_data.csv (single_skin or ram_air, ram_air when the
file or the row is missing).

The canopy of a single-skin wing has two kinds of cells:
    ramair  closed cells (here the tip cells): the lower skin covers the chord of the cell, the
            cell is inflated by the internal pressure P_vent of the json
    single  the other cells, leading edge included: every skin is a single foil loaded by the
            pressure difference between its two sides, without P_vent

The pieces of <wing>_canopy.csv name the cell and the skin of every canopy triangle:
    "<cell>.<k>:<name>T<j>_<side>"  upper skin of the cell
    "<cell>.<k>:<name>B<j>_<side>"  lower skin of the cell
    "<rib>:<rib>_<side>"            rib wall exposed to the flow (closes a ram air cell)
A cell (cell number + side l / r) is ram air when its lower skin covers at least LOWER_COVER of the
chordwise (x) extent of the cell in <wing>.msh. Exposed rib walls are ram air.

Inner surfaces (where an internal pressure would be collected) are not handled yet: none is listed
in the <wing>_data.csv of the single-skin wings so far.

Usage (from the tutorials directory): python single_skin.py <wing dir>   prints the zones of every cell
"""
import argparse
import collections
import csv
import os
import re

import numpy as np

from close_canopy import read_msh


LOWER_COVER = 0.8       # a cell is ram air when its lower skin covers this fraction of its chord

RAMAIR, SINGLE = "ramair", "single"
PIECE_RE = re.compile(r"^(\d+)\.\d+:.*?([TB])\d+_([lr])$")


def wing_type(wing_dir, name):
    """wing_type of <wing>_data.csv ("ram_air" when the file or the row is missing)."""
    path = os.path.join(wing_dir, name + "_data.csv")
    if not os.path.isfile(path):
        return "ram_air"
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["parameter"].strip() == "wing_type":
                return r["value"].strip()
    return "ram_air"


def is_single_skin(wing_dir, name):
    return wing_type(wing_dir, name) == "single_skin"


def read_pieces(wing_dir, name):
    """gmsh triangle number -> piece of <wing>_canopy.csv."""
    with open(os.path.join(wing_dir, name + "_canopy.csv"), newline="") as f:
        return {int(r["triangle"]): r["piece"].strip() for r in csv.DictReader(f)}


def piece_cell_skin(piece):
    """((cell, side), "T" upper / "B" lower) of a canopy piece, None for a rib wall."""
    m = PIECE_RE.match(piece)
    return ((m.group(1), m.group(3)), m.group(2)) if m else None


def canopy_zones(wing_dir, name, verbose=True):
    """{gmsh triangle number: RAMAIR or SINGLE} for every triangle of <wing>_canopy.csv, from the
    undeformed <wing>.msh, and {gmsh triangle number: "T" / "B" / None (rib wall)}."""
    pieces = read_pieces(wing_dir, name)
    _, nodes, elements = read_msh(os.path.join(wing_dir, name + ".msh"))
    conn = {e[0]: e[3] for e in elements if e[1] == 2}

    cell_x = collections.defaultdict(lambda: {"T": [], "B": []})
    skin = {}
    for tri, piece in pieces.items():
        if tri not in conn:
            raise ValueError("triangle %d of %s_canopy.csv is not a triangle of %s.msh" % (tri, name, name))
        cs = piece_cell_skin(piece)
        skin[tri] = cs[1] if cs else None
        if cs:
            cell_x[cs[0]][cs[1]].extend(nodes[i][0] for i in conn[tri])

    ram_cells = set()
    for cell in sorted(cell_x, key=lambda c: (c[1], int(c[0]))):
        x = cell_x[cell]
        all_x = x["T"] + x["B"]
        cover = np.ptp(x["B"]) / np.ptp(all_x) if x["B"] and np.ptp(all_x) > 0 else 0.0
        if cover >= LOWER_COVER:
            ram_cells.add(cell)
        if verbose:
            print("  cell %3s_%s: lower skin covers %3.0f%% of the chord -> %s"
                  % (cell[0], cell[1], 100 * cover, RAMAIR if cell in ram_cells else SINGLE))

    zones = {}
    for tri, piece in pieces.items():
        cs = piece_cell_skin(piece)
        zones[tri] = RAMAIR if cs is None or cs[0] in ram_cells else SINGLE
    if verbose:
        n = collections.Counter(zones.values())
        print("  %d ram air triangles (%d cells), %d single skin triangles"
              % (n[RAMAIR], len(ram_cells), n[SINGLE]))
    return zones, skin


def write_zones_csv(path, zones):
    with open(path, "w", newline="") as f:
        f.write("triangle,zone\n")
        f.writelines("%d,%s\n" % (tri, zone) for tri, zone in sorted(zones.items()))
    print("  written %s" % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wing", help="wing directory")
    args = parser.parse_args()
    wing_dir = os.path.normpath(args.wing)
    name = os.path.basename(wing_dir)
    print("%s: wing_type %s" % (name, wing_type(wing_dir, name)))
    canopy_zones(wing_dir, name)


if __name__ == "__main__":
    main()
