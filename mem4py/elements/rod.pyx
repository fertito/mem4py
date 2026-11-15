# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
"""
Discrete bending hinge for rods made of "B" elements.

A hinge sits on three consecutive rod nodes (x0, x1, x2). It carries only the
bending of the rod; the axial stiffness stays in the bar part of the "B" elements (same as "C").

The rod is stress-free when straight. With the turning angle phi between the
edges a = x1 - x0 and b = x2 - x1, the discrete curvature is
kappa = 2 tan(phi / 2) / l0 (discrete elastic rod), with l0 the mean of the two
initial edge lengths. The bending energy is

    U = EI kappa^2 l0 / 2 = kB * 2 tan^2(phi / 2) = 2 kB (1 - c) / (1 + c),

with kB = EI / l0 and c = cos(phi) = a.b / (|a||b|). The moment is
M = EI kappa. A rod meshed along a curve therefore starts with the bending
moment of a straight rod bent into that curve.

The internal force is the exact gradient of U. The stiffness written into the
sparse matrix (used by DR for mass scaling only) is the transverse bending
stiffness k_eff * w w^T (x) (I - t t^T) with w = [1/|a|, -(1/|a| + 1/|b|), 1/|b|],
t the mean tangent and k_eff = d2U/dphi2 = kB (1 + T^2)(1 + 3 T^2), T = tan(phi/2).
"""
cimport cython
from libc.math cimport sqrt


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int rodBendingFintAndK(double [:] X,
                            double [:] Y,
                            double [:] Z,
                            int [:, ::1] NBend,
                            Py_ssize_t el,
                            double kB,
                            unsigned int dim,
                            unsigned int index,
                            double [:] Fint,
                            double [:] data,
                            double [:] diagK,
                            unsigned int [:] order,
                            double * SE) nogil:
    """
    internal force, approximate stiffness and energy of one bending hinge
    :param NBend:   hinge connectivity [node 0, node 1 (middle), node 2]
    :param kB:      EI / l0
    :param dim:     2 or 3
    :param index:   start of this hinge's (3 dim)^2 block in the sparsity pattern
    :param SE:      bending energy U of the hinge
    :return: 0
    """
    cdef:
        double a[3]
        double b[3]
        double ah[3]
        double bh[3]
        double t[3]
        double g[3][3]
        double w[3]
        double la, lb, lt, c, opc, dUdc, T2, kEff, Kab, P
        Py_ssize_t n, m, i, j, dofA, ndofLoc = 3 * dim

    a[0] = X[NBend[el, 1]] - X[NBend[el, 0]]
    a[1] = Y[NBend[el, 1]] - Y[NBend[el, 0]]
    a[2] = Z[NBend[el, 1]] - Z[NBend[el, 0]]
    b[0] = X[NBend[el, 2]] - X[NBend[el, 1]]
    b[1] = Y[NBend[el, 2]] - Y[NBend[el, 1]]
    b[2] = Z[NBend[el, 2]] - Z[NBend[el, 1]]

    if dim == 2:
        a[2] = 0
        b[2] = 0

    la = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    lb = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])

    for i in range(3):
        ah[i] = a[i] / la
        bh[i] = b[i] / lb

    c = ah[0] * bh[0] + ah[1] * bh[1] + ah[2] * bh[2]

    # 1 + cos(phi), guarded against a fully folded hinge (phi = pi)
    opc = 1 + c
    if opc < 1e-8:
        opc = 1e-8

    # energy and its derivative with respect to c
    SE[0] = 2 * kB * (1 - c) / opc
    dUdc = -4 * kB / (opc * opc)

    # dc/dx for the three nodes: g[0] = -dc/da, g[1] = dc/da - dc/db, g[2] = dc/db
    for i in range(3):
        g[0][i] = -(bh[i] - c * ah[i]) / la
        g[2][i] = (ah[i] - c * bh[i]) / lb
        g[1][i] = -g[0][i] - g[2][i]

    # internal force Fint = dU/dx
    for n in range(3):
        for i in range(dim):
            Fint[dim * NBend[el, n] + i] += dUdc * g[n][i]

    # transverse bending stiffness for mass scaling
    T2 = (1 - c) / opc   # tan^2(phi / 2)
    kEff = kB * (1 + T2) * (1 + 3 * T2)

    w[0] = 1 / la
    w[2] = 1 / lb
    w[1] = -(w[0] + w[2])

    for i in range(3):
        t[i] = ah[i] + bh[i]
    lt = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
    if lt < 1e-12:
        for i in range(3):
            t[i] = ah[i]
    else:
        for i in range(3):
            t[i] /= lt

    # row major (3 dim) x (3 dim) block, local dof = node * dim + component
    for n in range(3):
        for i in range(dim):
            dofA = n * dim + i
            for m in range(3):
                for j in range(dim):
                    P = - t[i] * t[j]
                    if i == j:
                        P += 1
                    Kab = kEff * w[n] * w[m] * P
                    data[order[index + dofA * ndofLoc + m * dim + j]] += Kab
                    if n == m and i == j:
                        diagK[dim * NBend[el, n] + i] += Kab

    return 0
