"""
OpenFOAM flow around a closed paraglider wing (<name>_of.msh written by close_canopy.py), or
around the open canopy (--open).

Closed wing (default): the canopy + vent surface of the _of mesh is a closed (watertight) body. It
is written as an STL with two regions, meshed with blockMesh + snappyHexMesh (fluid outside the
wing only) and the steady incompressible RANS flow (foamRun, incompressibleFluid, k-omega SST,
SIMPLE) is solved at the requested speed and angle of attack. OpenFOAM-12 from blueCFD-Core 2024 is
used, in parallel with MS-MPI.

Open canopy (--open): only the "canopy" group of the mesh is used, without vent, as a zero-thickness
surface (snappyHexMesh faceZone with faceType baffle). The fluid is meshed outside and inside the
wing and the air enters the wing through the intakes. Every canopy face gets a wall face on each
side, in the patches wing and wing_slave (named after the surface). The mesh can be any gmsh
mesh with a "canopy" group, e.g. the original <wing>.msh.

Axes of the mesh: x chordwise (leading edge -> trailing edge), y spanwise, z up. The free stream
is U = speed * (cos(aoa), 0, sin(aoa)): a positive angle of attack is wind from below.

The finished case gets an empty case.foam file to open it in ParaView. Patches: wing_canopy and
wing_vent (wing and wing_slave with --open; group wingGroup), farfield.
Lift / drag coefficients (forceCoeffs function object,
reference area = wing area projected on the x-y plane, reference length = central chord) are in
postProcessing/forceCoeffs/.

Usage (from the tutorials directory):
    python wing_openfoam.py --aoa 8 --speed 10
    python wing_openfoam.py msh/demowing_of.msh --aoa 8 --speed 10 --iterations 2000 --procs 16
    python wing_openfoam.py --aoa 8 --speed 10 --no-run      write the case only
    python wing_openfoam.py msh/demowing.msh --open --aoa 8 --speed 10   open canopy
"""
import argparse
import glob
import os
import shutil
import subprocess
import time

import numpy as np

from close_canopy import read_msh


# --------------------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------------------

# OpenFOAM installation (blueCFD-Core 2024, OpenFOAM-12)
BLUECFD_DIR = r"C:\dev\blueCFD-Core-2024"
SETVARS = os.path.join(BLUECFD_DIR, "setvars_OF12.bat")
MPIEXEC = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"

# air (SI units, mesh is in m)
RHO_AIR = 1.225                 # [kg/m3] only used for the force coefficients
NU_AIR = 1.5E-5                 # [m2/s]
TURBULENCE_INTENSITY = 0.01
TURBULENCE_LENGTH = 0.1         # [m]

# domain, in wing spans b, measured from the wing bounding box
UPSTREAM, DOWNSTREAM, LATERAL, VERTICAL = 1.0, 3.0, 1.0, 1.0

# CFD mesh
BASE_CELL = 0.5                 # [m] background (blockMesh) cell size
WING_LEVEL = (4, 4)             # snappy refinement on the wing (cell = BASE_CELL / 2**level)
FEATURE_LEVEL = 5               # refinement on sharp edges (trailing edge, intake lips)
NEAR_LEVEL = 3                  # box around the wing
WAKE_LEVEL = 2                  # box around the wing and its wake
WAKE_LENGTH = 1.0               # [b] wake box length behind the trailing edge

# CFD run (steady SIMPLE)
ITERATIONS = 1000
WRITE_INTERVAL = 250

CASE_ROOT = "wing_openfoam"


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------

def foam_file(cls, obj, body):
    return ("FoamFile\n{\n    format      ascii;\n    class       %s;\n    object      %s;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "%s\n\n// ************************************************************************* //\n"
            % (cls, obj, body))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(text)


def stl_solid(name, P, tri):
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1)[:, None]
    lines = ["solid %s" % name]
    for k in range(len(tri)):
        lines.append("  facet normal %.9e %.9e %.9e" % tuple(n[k]))
        lines.append("    outer loop")
        for v in (a[k], b[k], c[k]):
            lines.append("      vertex %.9e %.9e %.9e" % tuple(v))
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid %s" % name)
    return "\n".join(lines) + "\n"


def processor_split(n):
    """(nx, ny, nz) with nx * ny * nz = n, as even as possible, most subdomains along x."""
    best = None
    for nx in range(1, n + 1):
        for ny in range(1, n // nx + 1):
            if n % (nx * ny) == 0:
                nz = n // (nx * ny)
                if nx >= ny >= nz:
                    score = nx - nz
                    if best is None or score < best[0]:
                        best = (score, (nx, ny, nz))
    return best[1]


def read_wing(mesh, groups=("canopy", "vent")):
    """Nodes and triangles of the given 2D groups of the mesh (canopy and vent of the _of mesh)."""
    names, nodes, elements = read_msh(mesh)
    tags = {name: tag for tag, (dim, name) in names.items() if dim == 2}
    for group in groups:
        if group not in tags:
            raise ValueError("no 2D physical group '%s' in %s (run close_canopy.py first)" % (group, mesh))
    ids = sorted(nodes)
    index = {i: k for k, i in enumerate(ids)}
    P = np.array([nodes[i] for i in ids])
    tri = {group: np.array([[index[i] for i in e[3]] for e in elements
                            if e[1] == 2 and e[2] and e[2][0] == tags[group]]).reshape(-1, 3)
           for group in groups}
    return P, tri


# --------------------------------------------------------------------------------------------------
# OpenFOAM case
# --------------------------------------------------------------------------------------------------

def write_case(case, P, tri, aoa, speed, iterations, n_procs, open_canopy=False):
    """tri: {"canopy": ..., "vent": ...} for the closed wing, {"canopy": ...} with open_canopy (the
    canopy is meshed as a zero-thickness baffle, fluid on both sides)."""
    all_tri = np.vstack(list(tri.values()))
    wing = P[np.unique(all_tri)]
    wmin, wmax = wing.min(axis=0), wing.max(axis=0)
    span = wmax[1] - wmin[1]

    # reference values: area projected on the x-y plane (half the |z| projection of the closed
    # surface, or of upper + lower skin of the open canopy), chord at the centre of the wing
    a, b, c = P[all_tri[:, 0]], P[all_tri[:, 1]], P[all_tri[:, 2]]
    Aref = 0.25 * np.abs(np.cross(b - a, c - a)[:, 2]).sum()
    centre = wing[np.abs(wing[:, 1] - 0.5 * (wmin[1] + wmax[1])) < 0.02 * span]
    lRef = centre[:, 0].max() - centre[:, 0].min() if len(centre) else wmax[0] - wmin[0]
    CofR = np.array([wmin[0] + 0.25 * lRef, 0.5 * (wmin[1] + wmax[1]), wmax[2]])

    alpha = np.radians(aoa)
    flow_dir = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    lift_dir = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])
    U_inf = speed * flow_dir

    # domain and background mesh
    lo = wmin - span * np.array([UPSTREAM, LATERAL, VERTICAL])
    hi = wmax + span * np.array([DOWNSTREAM, LATERAL, VERTICAL])
    n = np.maximum(np.ceil((hi - lo) / BASE_CELL).astype(int), 1)
    hi = lo + n * BASE_CELL
    location = lo + np.array([1.37, 1.41, 1.43]) * BASE_CELL     # a point in the fluid

    near_lo, near_hi = wmin - 0.1 * span, wmax + 0.1 * span
    wake_lo = wmin - 0.2 * span
    wake_hi = wmax + 0.2 * span + np.array([WAKE_LENGTH * span, 0, 0])

    vec = lambda v: "(%.6g %.6g %.6g)" % tuple(v)

    k_in = 1.5 * (speed * TURBULENCE_INTENSITY) ** 2
    omega_in = np.sqrt(k_in) / (0.09 ** 0.25 * TURBULENCE_LENGTH)

    write(os.path.join(case, "constant", "geometry", "wing.stl"),
          "".join(stl_solid(group, P, t) for group, t in tri.items()))

    # open canopy: zero-thickness surface, the cells on both sides are kept and the zone faces are
    # split into two walls (wing and wing_slave); the zone faces lie between cells of the same
    # (unnamed) cellZone, so free standing zone faces must be allowed
    baffle = "\n            faceZone wing;\n            faceType baffle;" if open_canopy else ""

    write(os.path.join(case, "system", "controlDict"), foam_file("dictionary", "controlDict", """\
application     foamRun;
solver          incompressibleFluid;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         %d;
deltaT          1;
writeControl    timeStep;
writeInterval   %d;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{
    forceCoeffs
    {
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         ("wing.*");
        rho             rhoInf;
        rhoInf          %g;
        CofR            %s;
        liftDir         %s;
        dragDir         %s;
        pitchAxis       (0 1 0);
        magUInf         %g;
        lRef            %.6g;
        Aref            %.6g;
    }
}""" % (iterations, min(WRITE_INTERVAL, iterations), RHO_AIR, vec(CofR), vec(lift_dir), vec(flow_dir),
        speed, lRef, Aref)))

    write(os.path.join(case, "system", "blockMeshDict"), foam_file("dictionary", "blockMeshDict", """\
convertToMeters 1;

vertices
(
    %s %s %s %s
    %s %s %s %s
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (%d %d %d) simpleGrading (1 1 1)
);

boundary
(
    farfield
    {
        type patch;
        faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (1 2 6 5) (2 3 7 6) (3 0 4 7));
    }
);""" % (vec((lo[0], lo[1], lo[2])), vec((hi[0], lo[1], lo[2])), vec((hi[0], hi[1], lo[2])),
         vec((lo[0], hi[1], lo[2])), vec((lo[0], lo[1], hi[2])), vec((hi[0], lo[1], hi[2])),
         vec((hi[0], hi[1], hi[2])), vec((lo[0], hi[1], hi[2])), n[0], n[1], n[2])))

    write(os.path.join(case, "system", "surfaceFeaturesDict"), foam_file("dictionary", "surfaceFeaturesDict", """\
surfaces ("wing.stl");

includedAngle       150;

subsetFeatures
{
    nonManifoldEdges    no;
    openEdges           yes;
}"""))

    write(os.path.join(case, "system", "snappyHexMeshDict"), foam_file("dictionary", "snappyHexMeshDict", """\
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    wing
    {
        type triSurfaceMesh;
        file "wing.stl";
    }
    nearBox
    {
        type searchableBox;
        min %s;
        max %s;
    }
    wakeBox
    {
        type searchableBox;
        min %s;
        max %s;
    }
};

castellatedMeshControls
{
    maxLocalCells 4000000;
    maxGlobalCells 40000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;

    features
    (
        {
            file "wing.eMesh";
            level %d;
        }
    );

    refinementSurfaces
    {
        wing
        {
            level (%d %d);
            patchInfo
            {
                type wall;
                inGroups (wingGroup);
            }%s
        }
    }

    resolveFeatureAngle 30;

    refinementRegions
    {
        wakeBox
        {
            mode    inside;
            level   %d;
        }
        nearBox
        {
            mode    inside;
            level   %d;
        }
    }

    insidePoint %s;

    allowFreeStandingZoneFaces %s;
}

snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers
    {
    }
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 100;
    slipFeatureAngle 30;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}

meshQualityControls
{
    #include "meshQualityDict"
}

mergeTolerance 1e-6;""" % (vec(near_lo), vec(near_hi), vec(wake_lo), vec(wake_hi), FEATURE_LEVEL,
                           WING_LEVEL[0], WING_LEVEL[1], baffle, WAKE_LEVEL, NEAR_LEVEL, vec(location),
                           "true" if open_canopy else "false")))

    write(os.path.join(case, "system", "meshQualityDict"), foam_file("dictionary", "meshQualityDict", """\
#includeEtc "caseDicts/mesh/generation/meshQualityDict"

minFaceWeight 0.02;"""))

    write(os.path.join(case, "system", "fvSchemes"), foam_file("dictionary", "fvSchemes", """\
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear limited corrected 0.5;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         limited corrected 0.5;
}

wallDist
{
    method meshWave;
}"""))

    write(os.path.join(case, "system", "fvSolution"), foam_file("dictionary", "fvSolution", """\
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }

    Phi
    {
        $p;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;

    residualControl
    {
        p               1e-4;
        U               1e-5;
        "(k|omega)"     1e-5;
    }
}

relaxationFactors
{
    equations
    {
        U               0.9;
        k               0.5;
        omega           0.5;
    }
}

potentialFlow
{
    nNonOrthogonalCorrectors 10;
}

cache
{
    grad(U);
}"""))

    write(os.path.join(case, "system", "decomposeParDict"), foam_file("dictionary", "decomposeParDict", """\
numberOfSubdomains  %d;

decomposer      hierarchical;
distributor     hierarchical;       // load balancing during parallel snappyHexMesh

hierarchicalCoeffs
{
    n               (%d %d %d);
    order           xyz;
}""" % ((n_procs,) + processor_split(n_procs))))

    write(os.path.join(case, "constant", "physicalProperties"), foam_file("dictionary", "physicalProperties", """\
viscosityModel  constant;

nu              [0 2 -1 0 0 0 0] %g;""" % NU_AIR))

    write(os.path.join(case, "constant", "momentumTransport"), foam_file("dictionary", "momentumTransport", """\
simulationType RAS;

RAS
{
    model               kOmegaSST;
    turbulence          on;
    printCoeffs         on;
}"""))

    # initial and boundary conditions: free stream on every outer face (in- or outflow depending
    # on the local flux, works for any angle of attack), walls on every wing* patch (both sides of
    # the open canopy baffle)
    walls = '"wing.*"'
    def field(cls, name, dims, internal, patches):
        body = "dimensions      %s;\n\ninternalField   %s;\n\nboundaryField\n{\n" \
               "    #includeEtc \"caseDicts/setConstraintTypes\"\n" % (dims, internal)
        for patch, entries in patches:
            body += "\n    %s\n    {\n%s    }\n" % (patch, "".join("        %s;\n" % e for e in entries))
        return foam_file(cls, name, body + "}")

    U_val = "uniform %s" % vec(U_inf)
    write(os.path.join(case, "0", "U"), field("volVectorField", "U", "[0 1 -1 0 0 0 0]", U_val, [
        ("farfield", ["type freestreamVelocity", "freestreamValue " + U_val, "value " + U_val]),
        (walls, ["type noSlip"])]))

    write(os.path.join(case, "0", "p"), field("volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0", [
        ("farfield", ["type freestreamPressure", "freestreamValue uniform 0", "value uniform 0"]),
        (walls, ["type zeroGradient"])]))

    k_val = "uniform %g" % k_in
    write(os.path.join(case, "0", "k"), field("volScalarField", "k", "[0 2 -2 0 0 0 0]", k_val, [
        ("farfield", ["type inletOutlet", "inletValue " + k_val, "value " + k_val]),
        (walls, ["type kqRWallFunction", "value " + k_val])]))

    w_val = "uniform %g" % omega_in
    write(os.path.join(case, "0", "omega"), field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", w_val, [
        ("farfield", ["type inletOutlet", "inletValue " + w_val, "value " + w_val]),
        (walls, ["type omegaWallFunction", "value " + w_val])]))

    write(os.path.join(case, "0", "nut"), field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", [
        ("farfield", ["type calculated", "value uniform 0"]),
        (walls, ["type nutkWallFunction", "value uniform 0"])]))

    # runner: sets up the blueCFD environment, then runs the given command in the case directory
    # setvars_OF12.bat puts lib\MS-MPI-10.1 on the PATH but the MPI build of libPstream is in
    # lib\MS-MPI-10.1.2, so OpenFOAM would load the serial (dummy) one: put the real one first
    mpi_lib = "".join(r'set "PATH=%%FOAM_LIBBIN%%\%s;%%PATH%%"' % os.path.basename(d) + "\r\n" for d in
                      glob.glob(os.path.join(BLUECFD_DIR, "OpenFOAM-12", "platforms", "*", "lib", "MS-MPI-*"))[:1])
    write(os.path.join(case, "run_foam.bat"),
          '@echo off\r\ncall "%s" >nul\r\n%scd /d "%s"\r\n%%*\r\nexit /b %%errorlevel%%\r\n'
          % (SETVARS, mpi_lib, os.path.abspath(case)))

    print("  wing: span %.3g m, central chord %.3g m, projected area %.3g m2" % (span, lRef, Aref))
    print("  free stream: %g m/s at %g deg -> U = %s, Re(chord) = %.3g"
          % (speed, aoa, vec(U_inf), speed * lRef / NU_AIR))
    print("  background mesh %d x %d x %d cells of %g m" % (n[0], n[1], n[2], BASE_CELL))


# --------------------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------------------

def run_foam(case, args, log, check=True):
    """Run an OpenFOAM command in the case through the blueCFD environment, output to log."""
    case = os.path.abspath(case)
    log_path = os.path.join(case, log)
    t0 = time.time()
    app = args[args.index("-genvlist") + 2:] if "-genvlist" in args else args   # skip mpiexec options
    print("    %-45s" % " ".join(app), end="", flush=True)
    with open(log_path, "w") as f:
        rc = subprocess.run(["cmd", "/c", os.path.join(case, "run_foam.bat")] + list(args),
                            stdout=f, stderr=subprocess.STDOUT).returncode
    print("%8.1f s  (%s)%s" % (time.time() - t0, log, "" if rc == 0 else "  exit code %d" % rc))
    if rc != 0 and check:
        with open(log_path, errors="replace") as f:
            tail = "".join(f.readlines()[-30:])
        raise RuntimeError("%s failed (exit code %d), see %s:\n%s" % (args[0], rc, log_path, tail))
    return rc


def run_case(case, n_procs):
    """Mesh the case and solve the flow (in parallel when n_procs > 1)."""
    run_foam(case, ["surfaceFeatures"], "log.surfaceFeatures")
    run_foam(case, ["blockMesh"], "log.blockMesh")

    if n_procs > 1:
        mpi = [MPIEXEC, "-n", str(n_procs), "-genvlist", "WM_PROJECT_DIR,MPI_BUFFER_SIZE,PATH"]
        run_foam(case, ["decomposePar", "-force"], "log.decomposePar")
        run_foam(case, mpi + ["snappyHexMesh", "-parallel", "-overwrite"], "log.snappyHexMesh")
        # snappy changed the mesh: put fresh initial fields in every processor directory
        for proc in glob.glob(os.path.join(case, "processor*")):
            shutil.rmtree(os.path.join(proc, "0"), ignore_errors=True)
            shutil.copytree(os.path.join(case, "0"), os.path.join(proc, "0"))
        run_foam(case, mpi + ["checkMesh", "-parallel"], "log.checkMesh", check=False)
        run_foam(case, mpi + ["potentialFoam", "-parallel"], "log.potentialFoam", check=False)
        run_foam(case, mpi + ["foamRun", "-parallel"], "log.foamRun")
        run_foam(case, ["reconstructPar"], "log.reconstructPar")   # OpenFOAM-12: mesh and fields
    else:
        run_foam(case, ["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
        run_foam(case, ["checkMesh"], "log.checkMesh", check=False)
        run_foam(case, ["potentialFoam"], "log.potentialFoam", check=False)
        run_foam(case, ["foamRun"], "log.foamRun")


def print_force_coeffs(case):
    files = sorted(glob.glob(os.path.join(case, "postProcessing", "forceCoeffs", "*", "forceCoeffs.dat")))
    if not files:
        print("  no force coefficients found (see log.foamRun)")
        return
    with open(files[-1]) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    header = [l for l in lines if l.startswith("#")][-1].lstrip("#").split()
    last = [l for l in lines if not l.startswith("#")][-1].split()
    print("  force coefficients at iteration %s: %s" % (last[0], ", ".join(
        "%s = %.4g" % (h, float(v)) for h, v in zip(header[1:], last[1:]) if h in ("Cd", "Cl", "Cm", "CmPitch"))))


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------

def run_wing(mesh, aoa, speed, case=None, iterations=ITERATIONS, n_procs=16, run=True, open_canopy=False):
    """Write (and run unless run=False) the OpenFOAM case of the closed wing mesh, or of the open
    canopy of the mesh (open_canopy); return the path of its case.foam. The case directory is
    deleted first if it exists."""
    if not open_canopy and not mesh.endswith("_of.msh"):
        print("Warning: %s does not end with _of.msh, the wing must be a closed canopy + vent surface" % mesh)

    name = os.path.splitext(os.path.basename(mesh))[0]
    case = case or os.path.join(CASE_ROOT, "%s_aoa%g_U%g" % (name, aoa, speed))
    if os.path.isdir(case):
        shutil.rmtree(case)

    start = time.time()
    print("Case %s (aoa %g deg, speed %g m/s, %s)"
          % (case, aoa, speed, "open canopy" if open_canopy else "closed wing"))
    if open_canopy:
        P, tri = read_wing(mesh, ("canopy",))
        print("  %s: %d canopy triangles (open, meshed as a baffle)" % (mesh, len(tri["canopy"])))
    else:
        P, tri = read_wing(mesh)
        print("  %s: %d canopy + %d vent triangles" % (mesh, len(tri["canopy"]), len(tri["vent"])))
    write_case(case, P, tri, aoa, speed, iterations, n_procs, open_canopy)

    foam = os.path.join(case, "case.foam")
    write(foam, "")

    if run:
        run_case(case, n_procs)
        print_force_coeffs(case)

    print("Finished in %.1f s" % (time.time() - start))
    print("Open in ParaView: %s" % os.path.abspath(foam))
    return foam


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", nargs="?", default=os.path.join("msh", "demowing_of.msh"),
                        help="closed wing mesh from close_canopy.py (default: msh/demowing_of.msh), "
                             "any mesh with a canopy group with --open")
    parser.add_argument("--aoa", type=float, required=True, help="angle of attack [deg]")
    parser.add_argument("--speed", type=float, required=True, help="free stream speed [m/s]")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="SIMPLE iterations (default %d)" % ITERATIONS)
    parser.add_argument("--procs", type=int, default=16, help="MPI processes (default 16, 1 = serial)")
    parser.add_argument("--case", help="case directory (default: %s/<mesh>_aoa<aoa>_U<speed>)" % CASE_ROOT)
    parser.add_argument("--no-run", action="store_true", help="only write the case")
    parser.add_argument("--open", action="store_true",
                        help="open canopy: the canopy group meshed as a zero-thickness baffle, no vent")
    args = parser.parse_args()
    run_wing(args.mesh, args.aoa, args.speed, args.case, args.iterations, args.procs, not args.no_run,
             args.open)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
