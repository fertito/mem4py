# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
import os
import numpy as np
cimport numpy as np
cimport cython

from helper.dirichletHandler cimport initialiseDirichletBC
from helper.area cimport area


cdef extern from "math.h":
    double sqrt(double m)
    double atan2(double m, double n)
    double cos(double m)
    double sin(double m)


cdef int readMsh(object data) except -1:
    """
    readMsh function reads .msh file and prepares nodal and element information
    :param data: mem4py main object with all necessary info
    :return: errorCode

    How the load definition work:

    constant load:

        - fX, fY, fZ
        - edgeX, edgeY, edgeZ
        - gravity gX, gY, gZ

    follower load:

        - p
        - edgeNormal

    node ID:
    fX = 1
    fY = 2
    fZ = 3

    line ID:
    "edgeX" = 1
    "edgeY" = 2
    "edgeZ" = 3
    "edgeNormal" = 4

    """
    # Initialise variables
    state = ""
    readArgs = True
    fixedBCNodesRead = []
    loadedBCNodesRead = []
    loadedBCEdgesRead = []
    loadedBCSurfaceRead = []
    prescribedBCNodesRead = []
    bnames = dict()
    X0read = []
    Y0read = []
    Z0read = []
    N1Read = []
    N2Read = []
    N3Read = []
    setID = 25
    pressureID = []
    edgeX_ID = []
    edgeY_ID = []
    edgeZ_ID = []
    FSIID = []
    elSet1 = []
    elSet2 = []
    elSet3 = []
    pressureMag = []
    edgeX_Mag = []
    edgeY_Mag = []
    edgeZ_Mag = []

    pre_stress_cable_ID = []
    pre_stress_cable_mag = []

    pre_strain_cable_ID = []
    pre_strain_cable_mag = []

    pre_stress_membrane_ID = []
    pre_stress_membrane_mag = []

    pre_strain_membrane_ID = []
    pre_strain_membrane_mag = []

    pre_u_read = []
    pre_v_read = []

    cdef:
        int nnodes = 0
        unsigned int i, j, k, q
        double f

    if data.restart is True:

        print('Restarting from {}.vtk'.format(data.restartName))

        read_u = False

        u_restart, v_restart, w_restart = [], [], []

        with open(data.restartName + '.vtk', 'r') as fin:

            for line in fin:

                args = line.split()

                if len(args) > 1:

                    if read_u is True:

                        u_restart.append(float(args[0]))
                        v_restart.append(float(args[1]))
                        w_restart.append(float(args[2]))

                    if args[1] == "u":

                        read_u = True

                elif read_u is True:

                    break

        u_restart = np.asarray(u_restart)
        v_restart = np.asarray(v_restart)
        w_restart = np.asarray(w_restart)

    with open(os.path.join(data.mshDir, "{}.msh".format(data.inputName))) as fin:
        for line in fin:
            if 'MeshFormat' in line:
                state = 'MeshFormat'
                readArgs = True
                continue
            elif 'PhysicalNames' in line:
                state = 'PhysicalNames'
                readArgs = True
                continue
            elif 'Nodes' in line:
                state = 'Nodes'
                readArgs = True
                continue
            elif 'Elements' in line:
                state = 'Elements'
                readArgs = True
                continue
            if readArgs:
                if state == 'MeshFormat':
                    args = line.split()
                    if float(args[0]) >= 4:
                        raise Exception("Cannot read msh input file version 4.0. Use \"-format msh2\" when creating msh file")
                    readArgs = False
                elif state == 'PhysicalNames':
                    readArgs = False
                    args = line.split()
                    IDmap = - np.ones(int(args[0]), dtype=np.intc)
                    continue
                elif state == 'Nodes':
                    nnodes = int(line)
                    readArgs = False
                    continue
                elif state == 'Elements':
                    nconditions = int(line)
                    readArgs = False
                    continue
            else:
                if state == 'PhysicalNames':
                    # dim, id, name
                    args = line.split()
                    name = args[2].replace('"', '')
                    bnames[int(args[1])] = name

                    # physical IDs need not be contiguous, grow the map to the largest ID
                    if int(args[1]) > len(IDmap):
                        IDmap = np.append(IDmap, - np.ones(int(args[1]) - len(IDmap), dtype=np.intc))

                    # lookup table format
                    # 0  -> M
                    # 1  -> MW
                    # 2  -> C
                    # 3  -> CW
                    # 2  -> B (C bar + rod bending hinges, see buildRodHinges)
                    # 4  -> fixAll
                    # 5  -> fixX
                    # 6  -> fixY
                    # 7  -> fixZ
                    # 8 ->  fixXY
                    # 9 ->  fixXZ
                    # 10 -> fixYZ
                    # 11 -> fX
                    # 12 -> fY
                    # 13 -> fZ
                    # 14 -> edgeX
                    # 15 -> edgeY
                    # 16 -> edgeZ
                    # 17 -> shearX
                    # 18 -> shearY
                    # 19 -> shearZ
                    # 20 -> edgeNormal
                    # 21 -> edgeShear
                    # 22 -> pressure
                    # 23 -> pre_u
                    # 24 -> pre_v
                    # >= 25 -> element set

                    if name not in data.elStruc:
                        raise Exception("Set {} not defined in msh file.".format(name))

                    # lookup table for element definitions
                    if data.elStruc[name]["set"] == "ELEMENT":

                        # cable and membrane pre_stress and pre-strain
                        if data.elStruc[name]["type"] in ("C", "CW", "B"):
                            if "pre_stress" in data.elStruc[name]:
                                pre_stress_cable_ID.append(int(args[1]))
                                pre_stress_cable_mag.append(data.elStruc[name]["pre_stress"])

                            if "pre_strain" in data.elStruc[name]:
                                pre_strain_cable_ID.append(int(args[1]))
                                pre_strain_cable_mag.append(data.elStruc[name]["pre_strain"])

                        elif data.elStruc[name]["type"] == "M" or data.elStruc[name]["type"] == "MW":
                            if "pre_stress" in data.elStruc[name]:
                                pre_stress_membrane_ID.append(int(args[1]))
                                pre_stress_membrane_mag.append(data.elStruc[name]["pre_stress"])

                            if "pre_strain" in data.elStruc[name]:
                                pre_strain_membrane_ID.append(int(args[1]))
                                pre_strain_membrane_mag.append(data.elStruc[name]["pre_strain"])

                        if "pressure" in data.elStruc[name]:
                            pressureID.append(int(args[1]))
                            pressureMag.append(data.elStruc[name]["pressure"])
                        if "edgeX" in data.elStruc[name]:
                            edgeX_ID.append(int(args[1]))
                            edgeX_Mag.append(data.elStruc[name]["edgeX"])
                        if "edgeY" in data.elStruc[name]:
                            edgeY_ID.append(int(args[1]))
                            edgeY_Mag.append(data.elStruc[name]["edgeY"])
                        if "edgeZ" in data.elStruc[name]:
                            edgeZ_ID.append(int(args[1]))
                            edgeZ_Mag.append(data.elStruc[name]["edgeZ"])
                        if "FSI" in data.elStruc[name]:
                            if data.elStruc[name]["FSI"] is True:
                                FSIID.append(int(args[1]))
                        if data.elStruc[name]["type"] == "M":
                            IDmap[int(args[1]) - 1] = 0
                        elif data.elStruc[name]["type"] == "MW":
                            IDmap[int(args[1]) - 1] = 1
                        elif data.elStruc[name]["type"] == "C" or data.elStruc[name]["type"] == "B":
                            IDmap[int(args[1]) - 1] = 2
                        elif data.elStruc[name]["type"] == "CW":
                            IDmap[int(args[1]) - 1] = 3
                        else:
                            raise Exception("No valid ELEMENT type found.")
                        if "EI" in data.elStruc[name] and data.elStruc[name]["type"] != "B":
                            raise Exception("Set {}: \"EI\" is only available for "
                                            "type \"B\" elements.".format(name))
                    elif data.elStruc[name]["set"] == "BC":
                        if data.elStruc[name]["type"] == "fixAll":
                            IDmap[int(args[1]) - 1] = 4
                        elif data.elStruc[name]["type"] == "fixX":
                            IDmap[int(args[1]) - 1] = 5
                        elif data.elStruc[name]["type"] == "fixY":
                            IDmap[int(args[1]) - 1] = 6
                        elif data.elStruc[name]["type"] == "fixZ":
                            IDmap[int(args[1]) - 1] = 7
                        elif data.elStruc[name]["type"] == "fixXY":
                            IDmap[int(args[1]) - 1] = 8
                        elif data.elStruc[name]["type"] == "fixXZ":
                            IDmap[int(args[1]) - 1] = 9
                        elif data.elStruc[name]["type"] == "fixYZ":
                            IDmap[int(args[1]) - 1] = 10
                        elif data.elStruc[name]["type"] == "pre_u":
                            IDmap[int(args[1]) - 1] = 23
                        else:
                            raise Exception("No valid BC type found.")
                    elif data.elStruc[name]["set"] == "LOAD":
                        if data.elStruc[name]["type"] == "fX":
                            IDmap[int(args[1]) - 1] = 11
                        elif data.elStruc[name]["type"] == "fY":
                            IDmap[int(args[1]) - 1] = 12
                        elif data.elStruc[name]["type"] == "fZ":
                            IDmap[int(args[1]) - 1] = 13
                        elif data.elStruc[name]["type"] == "edgeX":
                            IDmap[int(args[1]) - 1] = 14
                        elif data.elStruc[name]["type"] == "edgeY":
                            IDmap[int(args[1]) - 1] = 15
                        elif data.elStruc[name]["type"] == "edgeZ":
                            IDmap[int(args[1]) - 1] = 16
                        elif data.elStruc[name]["type"] == "shearX":
                            IDmap[int(args[1]) - 1] = 17
                        elif data.elStruc[name]["type"] == "shearY":
                            IDmap[int(args[1]) - 1] = 18
                        elif data.elStruc[name]["type"] == "shearZ":
                            IDmap[int(args[1]) - 1] = 19
                        elif data.elStruc[name]["type"] == "edgeNormal":
                            IDmap[int(args[1]) - 1] = 20
                        elif data.elStruc[name]["type"] == "edgeShear":
                            IDmap[int(args[1]) - 1] = 21
                        elif data.elStruc[name]["type"] == "aero":
                            IDmap[int(args[1]) - 1] = 22
                        elif data.elStruc[name]["type"] == "pressure":
                            IDmap[int(args[1]) - 1] = 23
                            pressureID.append(int(args[1]))
                            pressureMag.append(data.elStruc[name]["pressure"])
                        elif data.elStruc[name]["type"] == "damper":
                            IDmap[int(args[1]) - 1] = 24
                        else:
                            raise Exception("No valid LOAD type found.")
                    elif data.elStruc[name]["set"] == "SET":
                        IDmap[int(args[1]) - 1] = setID
                        setID += 1
                    else:
                        raise Exception("No set in elStruct found.")

                elif state == 'Nodes':
                    # Nodal coordinates in vectors X, Y, Z
                    args = line.split()
                    X0read.append(float(args[1]) * data.scale)
                    Y0read.append(float(args[2]) * data.scale)
                    Z0read.append(float(args[3]) * data.scale)
                elif state == 'Elements':
                    # elementNumber, elementType, numberOfTags, physicalID, elementaryID, nodeCon
                    #
                    # elementType:
                    # 1 = 2-node line
                    # 2 = 3-node triangle
                    # 8 = 3-node second order line
                    # 9 = 6-node second order triangle
                    # 15 = 1-node point
                    #
                    # numberOfTags:
                    # gives the number of integer tags that follow for current element.
                    # - First tag is always the physical entity (in this case the real element description as
                    # defined in bnames
                    # - Second tag is the elementary tag (currently not used)
                    #
                    # Last entries give node connectivity of element

                    args = line.split()

                    # 1 node element
                    if int(args[1]) == 15:

                        # [element ID, node ID]
                        j = IDmap[int(args[3]) - 1]
                        i = int(args[5]) - 1

                        if j > 24:
                            # save [physical ID, node number]
                            elSet1.append([int(args[3]), i])
                        elif j == 4:  # fixAll
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(i)
                        elif j == 5:  # fixX
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(i)
                        elif j == 6:  # fixY
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(i)
                        elif j == 7:  # fixZ
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(i)
                        elif j == 8:  # fixXY
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(i)
                        elif j == 9:  # fixXZ
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(i)
                        elif j == 10:  # fixYZ
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(i)

                        elif j == 11:  # fX
                            f = data.elStruc[bnames[int(args[3])]]["fX"]
                            loadedBCNodesRead.append([1, i, f])
                        elif j == 12:  # fY
                            f = data.elStruc[bnames[int(args[3])]]["fY"]
                            loadedBCNodesRead.append([2, i, f])
                        elif j == 13:  # fZ
                            f = data.elStruc[bnames[int(args[3])]]["fZ"]
                            loadedBCNodesRead.append([3, i, f])
                        # elif j == 14:  # edgeX
                        #     f = data.elStruc[bnames[int(args[3])]]["edgeX"]
                        #     loadedBCEdgesRead.append([1, i, f])
                        # elif j == 15:  # edgeY
                        #     f = data.elStruc[bnames[int(args[3])]]["edgeY"]
                        #     loadedBCEdgesRead.append([2, i, f])
                        # elif j == 16:  # edgeZ
                        #     f = data.elStruc[bnames[int(args[3])]]["edgeZ"]
                        #     loadedBCEdgesRead.append([3, i, f])
                        # elif j == 17:  # FSI
                        #     loadedBCEdgesRead.append([4, i, 0])
                        elif j == 21:  # pre_u
                            if "pre_u" in data.elStruc[bnames[int(args[3])]].keys():
                                pre_u_read.append([i,
                                                   data.elStruc[bnames[int(args[3])]]["pre_u"]])
                            if "pre_v" in data.elStruc[bnames[int(args[3])]].keys():
                                pre_v_read.append([i,
                                                   data.elStruc[bnames[int(args[3])]]["pre_v"]])
                        elif j == 22:  # damper
                            f = data.elStruc[bnames[int(args[3])]]["damper"]  # D * V**2, define D
                            mass = data.elStruc[bnames[int(args[3])]]["mass"]
                            loadedBCNodesRead.append([4, i, f])
                            loadedBCNodesRead.append([5, i, mass])

                    # 2 node element
                    elif int(args[1]) == 1:

                        # [physical name ID, node ID 1, node ID 2]
                        j = IDmap[int(args[3]) - 1]
                        i = int(args[5]) - 1
                        k = int(args[6]) - 1

                        if j > 24:
                            # save [pysical ID, node 1, node 2]
                            elSet2.append([int(args[3]), i, k])
                        elif j == 2 or j == 3:  # Cable element
                            N2Read.append([j, i, k, int(args[3])])
                        elif j == 4:  # fixAll
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(k)
                        elif j == 5:  # fixX
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(k)
                        elif j == 6:  # fixY
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(k)
                        elif j == 7:  # fixZ
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(k)
                        elif j == 8:  # fixXY
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(k)
                        elif j == 9:  # fixXZ
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(k)
                        elif j == 10:  # fixYZ
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(k)

                        if j == 11:  # fX
                            f = data.elStruc[bnames[int(args[3])]]["fX"]
                            loadedBCNodesRead.append([1, i, f])
                            loadedBCNodesRead.append([1, k, f])
                        if j == 12:  # fY
                            f = data.elStruc[bnames[int(args[3])]]["fY"]
                            loadedBCNodesRead.append([2, i, f])
                            loadedBCNodesRead.append([2, k, f])
                        if j == 13:  # fZ
                            f = data.elStruc[bnames[int(args[3])]]["fZ"]
                            loadedBCNodesRead.append([3, i, f])
                            loadedBCNodesRead.append([3, k, f])
                        if j == 14:  # edgeX
                            f = data.elStruc[bnames[int(args[3])]]["edgeX"]
                            loadedBCEdgesRead.append([1, i, k, f])
                        if j == 15:  # edgeY
                            f = data.elStruc[bnames[int(args[3])]]["edgeY"]
                            loadedBCEdgesRead.append([2, i, k, f])
                        if j == 16:  # edgeZ
                            f = data.elStruc[bnames[int(args[3])]]["edgeZ"]
                            loadedBCEdgesRead.append([3, i, k, f])
                        if j == 17:  # shearX
                            f = data.elStruc[bnames[int(args[3])]]["shearX"]
                            loadedBCEdgesRead.append([4, i, k, f])
                        if j == 18:  # shearY
                            f = data.elStruc[bnames[int(args[3])]]["shearY"]
                            loadedBCEdgesRead.append([5, i, k, f])
                        if j == 19:  # shearZ
                            f = data.elStruc[bnames[int(args[3])]]["shearZ"]
                            loadedBCEdgesRead.append([6, i, k, f])
                        if j == 20:  # edgeNormal
                            f = data.elStruc[bnames[int(args[3])]]["edgeNormal"]
                            loadedBCEdgesRead.append([7, i, k, f])
                        if j == 21:  # edgeShear
                            f = data.elStruc[bnames[int(args[3])]]["edgeShear"]
                            loadedBCEdgesRead.append([8, i, k, f])
                        if j == 22:  # aero
                            f = data.elStruc[bnames[int(args[3])]]["aero"]
                            loadedBCEdgesRead.append([9, i, k, f])
                        if j == 23:  # FSI
                            loadedBCEdgesRead.append([10, i, k, 0])
                        if j == 24:  # pre_u
                            if "pre_u" in data.elStruc[bnames[int(args[3])]].keys():
                                pre_u_read.append([i,
                                                   data.elStruc[bnames[int(args[3])]]["pre_u"]])
                                pre_u_read.append([k,
                                                   data.elStruc[bnames[int(args[3])]]["pre_u"]])
                            if "pre_v" in data.elStruc[bnames[int(args[3])]].keys():
                                pre_v_read.append([i,
                                                   data.elStruc[bnames[int(args[3])]]["pre_v"]])
                                pre_v_read.append([k,
                                                   data.elStruc[bnames[int(args[3])]]["pre_v"]])

                    # 3 node element
                    elif int(args[1]) == 2:

                        # [physical name ID, node ID 1, node ID 2, node ID 3]
                        j = IDmap[int(args[3]) - 1]
                        i = int(args[5]) - 1
                        k = int(args[6]) - 1
                        q = int(args[7]) - 1

                        if j > 24:
                            # save [physical ID, node 1, node 2, node 3]
                            elSet3.append([int(args[3]), i, k, q])
                        elif j == 0 or j == 1:  # M == 0, MW == 1
                            N3Read.append([j, i, k, q, int(args[3])])
                        elif j == 4:  # fixAll
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(q)
                        elif j == 5:  # fixX
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(q)
                        elif j == 6:  # fixY
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(q)
                        elif j == 7:  # fixZ
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(q)
                        elif j == 8:  # fixXY
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(q)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(q)
                        elif j == 9:  # fixXZ
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(q)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(q)
                        elif j == 10:  # fixYZ
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(q)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(i)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(k)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(q)
                        if j == 14:  # edgeX
                            f = data.elStruc[bnames[int(args[3])]]["edgeX"]
                            loadedBCSurfaceRead.append([1, i, k, q, f])
                        if j == 15:  # edgeY
                            f = data.elStruc[bnames[int(args[3])]]["edgeY"]
                            loadedBCSurfaceRead.append([2, i, k, q, f])
                        if j == 16:  # edgeZ
                            f = data.elStruc[bnames[int(args[3])]]["edgeZ"]
                            loadedBCSurfaceRead.append([3, i, k, q, f])
                        if j == 17:  # shearX
                            f = data.elStruc[bnames[int(args[3])]]["shearX"]
                            loadedBCSurfaceRead.append([4, i, k, q, f])
                        if j == 18:  # shearY
                            f = data.elStruc[bnames[int(args[3])]]["shearY"]
                            loadedBCSurfaceRead.append([5, i, k, q, f])
                        if j == 19:  # shearZ
                            f = data.elStruc[bnames[int(args[3])]]["shearZ"]
                            loadedBCSurfaceRead.append([6, i, k, q, f])
                        # if j == 20:  # edgeNormal
                        #     f = data.elStruc[bnames[int(args[3])]]["edgeNormal"]
                        #     loadedBCSurfaceRead.append([7, i, k, q, f])
                        # if j == 21:  # edgeShear
                        #     f = data.elStruc[bnames[int(args[3])]]["edgeShear"]
                        #     loadedBCSurfaceRead.append([8, i, k, q, f])
                        elif j == 22:  # pressure
                            N3Read.append([j, i, k, q, int(args[3])])

                    else:
                        raise Exception("Element identifier other than 15, 1, 2 found in element ID {}".format(int(args[0])))

    # Define connectivity matrices
    N1p = np.asarray(N1Read, dtype=np.intc)
    N2p = np.asarray(N2Read, dtype=np.intc)
    N3p = np.asarray(N3Read, dtype=np.intc)

    if N1p.size == 0:
        N1p = np.array([[-1, 0]], dtype=np.intc)
    if N2p.size == 0:
        N2p = np.array([[-1, 0, 0, 0]], dtype=np.intc)
    if N3p.size == 0:
        N3p = np.array([[-1, 0, 0, 0, 0]], dtype=np.intc)

    # assemble NMem and NCable
    cdef:
        unsigned int el = 0

        unsigned int nM = abs(len(np.where(N3p[:, 0] == 0)[0]))
        unsigned int nMW = abs(len(np.where(N3p[:, 0] == 1)[0]))
        unsigned int nC = abs(len(np.where(N2p[:, 0] == 2)[0]))
        unsigned int nCW = abs(len(np.where(N2p[:, 0] == 3)[0]))

        double [:] X0 = np.asarray(X0read)
        double [:] Y0 = np.asarray(Y0read)
        double [:] Z0 = np.asarray(Z0read)

        int [:, ::1] N1 = N1p
        int [:, ::1] N2 = N2p
        int [:, ::1] N3 = N3p  # N3 = [lookup table ID, N1, N2, N3, physical ID from gmsh]

        unsigned int nelemsMem = abs(nM + nMW)
        unsigned int nelemsCable = abs(nC + nCW)
        unsigned int nelems = abs(nelemsCable + nelemsMem)

    if nelemsMem == 0:
        nelemsMemTemp = 1
    else:
        nelemsMemTemp = nelemsMem
    if nelemsCable == 0:
        nelemsCableTemp = 1
    else:
        nelemsCableTemp = nelemsCable

    cdef:
        int [:, ::1] Nc = np.empty((nelemsCableTemp, 4), dtype=np.intc)
        int [:, ::1] Nm = np.empty((nelemsMemTemp, 5), dtype=np.intc)

    # edge loads on C2 elements
    for i in range(len(edgeX_ID)):
        ind = np.where(np.asarray(N2)[:, 3] == edgeX_ID[i])[0]
        for j in ind:
            loadedBCEdgesRead.append([1, N2[j, 1], N2[j, 2], edgeX_Mag[int(i)]])

    for i in range(len(edgeY_ID)):
        ind = np.where(np.asarray(N2)[:, 3] == edgeY_ID[i])[0]
        for j in ind:
            loadedBCEdgesRead.append([2, N2[j, 1], N2[j, 2], edgeY_Mag[int(i)]])

    for i in range(len(edgeZ_ID)):
        ind = np.where(np.asarray(N2)[:, 3] == edgeZ_ID[i])[0]
        for j in ind:
            loadedBCEdgesRead.append([3, N2[j, 1], N2[j, 2], edgeZ_Mag[int(i)]])

    # search cable elements
    for i in range(np.size(N2, 0)):

        # if element == C (bar element)
        if N2[i, 0] == 2:

            Nc[el, 0] = N2[i, 3]
            Nc[el, 1] = N2[i, 1]
            Nc[el, 2] = N2[i, 2]
            Nc[el, 3] = 0
            el += 1

        # if element == CW (cable element without compressive resistance)
        elif N2[i, 0] == 3:

            Nc[el, 0] = N2[i, 3]
            Nc[el, 1] = N2[i, 1]
            Nc[el, 2] = N2[i, 2]
            Nc[el, 3] = 1
            el += 1

    # edge loads on M3 elements
    for i in range(len(edgeX_ID)):
        ind = np.where(np.asarray(N3)[:, 4] == edgeX_ID[i])[0]
        for j in ind:
            loadedBCSurfaceRead.append([1, N3[j, 1], N3[j, 2], N3[j, 2], edgeX_Mag[int(i)]])

    for i in range(len(edgeY_ID)):
        ind = np.where(np.asarray(N3)[:, 4] == edgeY_ID[i])[0]
        for j in ind:
            loadedBCSurfaceRead.append([2, N3[j, 1], N3[j, 2], N3[j, 2], edgeY_Mag[int(i)]])

    for i in range(len(edgeZ_ID)):
        ind = np.where(np.asarray(N3)[:, 4] == edgeZ_ID[i])[0]
        for j in ind:
            loadedBCSurfaceRead.append([3, N3[j, 1], N3[j, 2], N3[j, 2], edgeZ_Mag[int(i)]])

    # find number of pressurised elements and corresponding element IDs
    elPressurisedTemp = []

    el = 0
    # search for membrane elements
    for i in range(np.size(N3, 0)):

        # if element == M (membrane)
        if N3[i, 0] == 0:

            Nm[el, 0] = N3[i, 4]
            Nm[el, 1] = N3[i, 1]
            Nm[el, 2] = N3[i, 2]
            Nm[el, 3] = N3[i, 3]
            Nm[el, 4] = 0
            el += 1

        # if element == MW (membrane with active wrinkling model)
        elif N3[i, 0] == 1:

            Nm[el, 0] = N3[i, 4]
            Nm[el, 1] = N3[i, 1]
            Nm[el, 2] = N3[i, 2]
            Nm[el, 3] = N3[i, 3]
            Nm[el, 4] = 1
            el += 1

        elif N3[i, 0] == 20:  # pressurized element

            # TODO: this could cause a seg fault...
            # find pressurized element in Nm and save its element ID
            if N3[i - 1, 1] == N3[i, 1] and N3[i - 1, 2] == N3[i, 2] and N3[i - 1, 3] == N3[i, 3]:
                elPressurisedTemp.append([N3[i, 4], el - 1])
            else:
                raise Exception("Previous element not the same as pressurized element.")

    # set arrays
    if len(elSet1) == 0:
        N1set = np.array([[-1, -1]], dtype=np.intc)
    else:
        N1set = np.asarray(elSet1, dtype=np.intc)

    if len(elSet2) == 0:
        N2set = np.array([[-1, -1]], dtype=np.intc)
    else:
        N2set = np.asarray(elSet2, dtype=np.intc)

    if len(elSet3) == 0:
        N3set = np.array([[-1, -1]], dtype=np.intc)
    else:
        N3set = np.asarray(elSet3, dtype=np.intc)

    cdef:
        unsigned int nPressurised = 0, nFSI = 0

    if pressureID:
        for i in pressureID:
            ind = np.where(np.asarray(Nm)[:, 0] == i)[0]
            for j in range(len(ind)):
                elPressurisedTemp.append([Nm[ind[j], 0], ind[j]])

    nPressurised = np.size(elPressurisedTemp, 0)

    if len(elPressurisedTemp) == 0:
        elPressurisedTemp = [[0, 0]]

    elPressurisedTemp = np.asarray(elPressurisedTemp, dtype=np.uintc)

    # find number of FSI elements and corresponding element IDs
    elFSITemp = []
    nFSIArray = []

    for i in FSIID:
        ind = np.where(np.asarray(Nm)[:, 0] == i)[0]
        nFSIArray.append(len(ind))
        nFSI += len(ind)
        elFSITemp.append(ind)

    if len(elFSITemp) == 0:
        elFSITemp = [0]
        nFSIArray = 0
    else:
        # flatten
        elFSITemp = [item for sublist in elFSITemp for item in sublist]

    elFSITemp = np.asarray(elFSITemp, dtype=np.uintc).flatten()
    nFSIArray = np.asarray(nFSIArray)

    # Fixed nodes as C array
    homogeneous_BC = True
    if not fixedBCNodesRead:
        homogeneous_BC = False
        fixedBCNodesRead = [0]

    cdef:
        unsigned int [:] fixedBCNodes = np.asarray(fixedBCNodesRead, dtype=np.uintc)
        int [:, ::1] N1Set = N1set
        int [:, ::1] N2Set = N2set
        int [:, ::1] N3Set = N3set
        unsigned int n_homogeneous_BC = 0

    if homogeneous_BC is True:
        # Call dirichletHandler for dofFree
        dofFixedP = initialiseDirichletBC(fixedBCNodes, data.dim)
        n_homogeneous_BC = len(dofFixedP)
    else:
        dofFixedP = np.array([0])
        n_homogeneous_BC = 0

    if not loadedBCNodesRead:
        loadedBCNodesRead = np.zeros((1, 3), dtype=np.double)
        loadedBCNodesRead_damper = np.zeros((1, 4), dtype=np.double)
    else:
        loadedBCNodesRead = np.asarray(loadedBCNodesRead, dtype=np.double)

        # sort for node IDs and remove
        ind_x = np.where(loadedBCNodesRead[:, 0] == 1)[0]
        ind_y = np.where(loadedBCNodesRead[:, 0] == 2)[0]
        ind_z = np.where(loadedBCNodesRead[:, 0] == 3)[0]
        ind_damper = np.where(loadedBCNodesRead[:, 0] == 4)[0]
        ind_mass = np.where(loadedBCNodesRead[:, 0] == 5)[0]

        if ind_damper.size:
            loadedBCNodesRead_damper = np.zeros((len(ind_damper), 4))
            loadedBCNodesRead_damper[:, 0:3] = loadedBCNodesRead[ind_damper, :]
            loadedBCNodesRead_damper[:, 3] = loadedBCNodesRead[ind_mass, 2]
        else:
            loadedBCNodesRead_damper = np.array([], dtype=np.double).reshape(0,3)

        if ind_x.size:
            loadedBCNodesReadX = np.unique(loadedBCNodesRead[ind_x, :], axis=0)
        else:
            loadedBCNodesReadX = np.array([], dtype=np.double).reshape(0,3)

        if ind_y.size:
            loadedBCNodesReadY = np.unique(loadedBCNodesRead[ind_y, :], axis=0)
        else:
            loadedBCNodesReadY = np.array([], dtype=np.double).reshape(0,3)

        if ind_z.size:
            loadedBCNodesReadZ = np.unique(loadedBCNodesRead[ind_z, :], axis=0)
        else:
            loadedBCNodesReadZ = np.array([], dtype=np.double).reshape(0,3)

        loadedBCNodesRead = np.concatenate([loadedBCNodesReadX,
                                            loadedBCNodesReadY,
                                            loadedBCNodesReadZ])

    cdef double [:, ::1] loadedBCNodes = np.asarray(loadedBCNodesRead, dtype=np.double)
    cdef double [:, ::1] loadedBCNodes_damper = np.asarray(loadedBCNodesRead_damper, dtype=np.double)
    data.loadedBCNodes_damper = loadedBCNodes_damper

    if not loadedBCEdgesRead:
        loadedBCEdgesRead = np.zeros((1, 4), dtype=np.double)
    else:
        loadedBCEdgesRead = np.asarray(loadedBCEdgesRead, dtype=np.double)
        loadedBCEdgesRead = loadedBCEdgesRead[loadedBCEdgesRead[:,0].argsort()]
    cdef double [:, ::1] loadedBCEdges = np.asarray(loadedBCEdgesRead, dtype=np.double)

    if not loadedBCSurfaceRead:
        loadedBCSurfaceRead = np.zeros((1, 5), dtype=np.double)
    else:
        loadedBCSurfaceRead = np.asarray(loadedBCSurfaceRead, dtype=np.double)
        loadedBCSurfaceRead = loadedBCSurfaceRead[loadedBCSurfaceRead[:,0].argsort()]
    cdef double [:, ::1] loadedBCSurface = np.asarray(loadedBCSurfaceRead, dtype=np.double)

    if prescribedBCNodesRead:
        prescribedBCNodesRead = np.unique(prescribedBCNodesRead, axis=0)
    else:
        prescribedBCNodesRead = - np.ones((1, 1), dtype=np.intc)

    cdef int [:] prescribedDof = np.empty(prescribedBCNodesRead.shape[0], dtype=np.intc)
    cdef double [:] prescribedDisplacement = np.zeros(prescribedBCNodesRead.shape[0], dtype=np.double)

    if prescribedBCNodesRead[0, 0] == -1:
        prescribedDof[0] = -1
    else:
        for el in range(prescribedBCNodesRead.shape[0]):
            if prescribedBCNodesRead[el, 0] == 1:
                prescribedDof[el] = data.dim * (prescribedBCNodesRead[el, 1] + 1) - data.dim
            elif prescribedBCNodesRead[el, 0] == 2:
                prescribedDof[el] = data.dim * (prescribedBCNodesRead[el, 1] + 1) - data.dim + 1
            else:
                prescribedDof[el] = data.dim * (prescribedBCNodesRead[el, 1] + 1) - data.dim + 2
        ind = np.argsort(prescribedDof)

        for el in range(len(ind)):
            prescribedDisplacement[el] = prescribedBCNodesRead[ind[el], 0]
        prescribedDof = np.sort(prescribedDof)
    
    cdef:
        int [:] dofFixed = dofFixedP.astype(np.intc)

        unsigned int [:] elPressurised = elPressurisedTemp[:, 1]
        unsigned int [:] elFSI = elFSITemp
        double [:] p = np.empty(nPressurised, dtype=np.double)

        double [:] Sx = np.zeros(len(elFSI), dtype=np.double)
        double [:] Sy = np.zeros(len(elFSI), dtype=np.double)
        double [:] Sz = np.zeros(len(elFSI), dtype=np.double)
        double [:] pFSI = np.zeros(len(elFSI), dtype=np.double)
    
    for i in range(nPressurised):
        j = np.where(pressureID == elPressurisedTemp[i, 0])[0]
        p[i] = pressureMag[j]

    # cell centres
    if data.dim == 2:
        el = nelemsCable
    elif data.dim == 3:
        el = len(elFSI)

    cdef double [:, ::1] cc = np.zeros((el, data.dim), dtype=np.double)

    # area
    cdef double [:] area3 = np.zeros(nelemsMemTemp, dtype=np.double)
    area(X0, Y0, Z0, Nm, nelemsMem, area3)
    data.area3 = area3

    # determine parts of the deformation gradient
    cdef:
        long double [:] E3 = np.zeros(3, dtype=np.longdouble)
        long double [:] E2 = np.zeros(3, dtype=np.longdouble)
        long double [:] E1 = np.zeros(3, dtype=np.longdouble)
        long double [:] normalVecX = np.zeros(nelemsMemTemp, dtype=np.longdouble)
        long double [:] normalVecY = np.zeros(nelemsMemTemp, dtype=np.longdouble)
        long double [:] normalVecZ = np.zeros(nelemsMemTemp, dtype=np.longdouble)
        long double [:] J11Vec = np.zeros(nelemsMemTemp, dtype=np.longdouble)
        long double [:] J12Vec = np.zeros(nelemsMemTemp, dtype=np.longdouble)
        long double [:] J22Vec = np.zeros(nelemsMemTemp, dtype=np.longdouble)
        double [:] thetaVec = np.zeros(nelemsMemTemp, dtype=np.double)
        double [:] thetaCylinder = np.zeros(nelemsMemTemp, dtype=np.double)
        long double Nnorm

        double [:] LCable = np.empty(nelemsCableTemp, dtype=np.double)
        double [:] ed = np.empty(3, dtype=np.double)
        double [:, ::1] R = np.empty((3, 3), dtype=np.double)

    # Material orientation
    matDir = data.matDir
    if data.matDirAngle:
        matDirAngle = data.matDirAngle
    else:
        matDirAngle = 0

    # cables
    for el in range(nelemsCable):

        LCable[el] = sqrt((X0[Nc[el, 2]] - X0[Nc[el, 1]]) * (X0[Nc[el, 2]] - X0[Nc[el, 1]]) +
                          (Y0[Nc[el, 2]] - Y0[Nc[el, 1]]) * (Y0[Nc[el, 2]] - Y0[Nc[el, 1]]) +
                          (Z0[Nc[el, 2]] - Z0[Nc[el, 1]]) * (Z0[Nc[el, 2]] - Z0[Nc[el, 1]]))

    # membranes
    for el in range(nelemsMem):

        J11Vec[el] = sqrt((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (X0[Nm[el, 2]] - X0[Nm[el, 1]]) +
                          (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) +
                          (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]))
        J12Vec[el] = ((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]]) +
                      (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) +
                      (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]])) / J11Vec[el]

        Nnorm = sqrt(((Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]]) -
                      (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]])) *
                     ((Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]]) -
                      (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]])) +

                     ((Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]]) -
                      (X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]])) *
                     ((Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]]) -
                      (X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]])) +

                     ((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) -
                      (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]])) *
                     ((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) -
                      (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]])))

        normalVecX[el] = ((Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]]) -
                      (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]])) / Nnorm
        normalVecY[el] = ((Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]]) -
                      (X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]])) / Nnorm
        normalVecZ[el] = ((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) -
                      (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]])) / Nnorm

        # unit vector of X12
        E1[0] = (X0[Nm[el, 2]] - X0[Nm[el, 1]]) / J11Vec[el]
        E1[1] = (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) / J11Vec[el]
        E1[2] = (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) / J11Vec[el]

        # normal (unit) vector to triangle surface
        E3[0] = ((Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]]) -
                (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]])) / (2 * area3[el])
        E3[1] = ((Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]]) -
                (X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]])) / (2 * area3[el])
        E3[2] = ((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) -
                (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]])) / (2 * area3[el])

        # unit vector (N x E1)
        E2[0] = E3[1] * E1[2] - E3[2] * E1[1]
        E2[1] = E3[2] * E1[0] - E3[0] * E1[2]
        E2[2] = E3[0] * E1[1] - E3[1] * E1[0]

        J22Vec[el] = (X0[Nm[el, 3]] - X0[Nm[el, 1]]) * E2[0] + \
                     (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) * E2[1] + \
                     (Z0[Nm[el, 3]] - Z0[Nm[el, 1]]) * E2[2]

        # print("J22Vec[el] = {}".format(J22Vec[el]))
        # test = sqrt(((Y0[N[el, 2]] - Y0[N[el, 1]]) * (Z0[N[el, 3]] - Z0[N[el, 1]]) -
        #              (Z0[N[el, 2]] - Z0[N[el, 1]]) * (Y0[N[el, 3]] - Y0[N[el, 1]])) ** 2 +
        # ((Z0[N[el, 2]] - Z0[N[el, 1]]) * (X0[N[el, 3]] - X0[N[el, 1]]) -
        #              (X0[N[el, 2]] - X0[N[el, 1]]) * (Z0[N[el, 3]] - Z0[N[el, 1]])) ** 2 +
        # ((X0[N[el, 2]] - X0[N[el, 1]]) * (Y0[N[el, 3]] - Y0[N[el, 1]]) -
        #              (Y0[N[el, 2]] - Y0[N[el, 1]]) * (X0[N[el, 3]] - X0[N[el, 1]])) ** 2) / J11Vec[el]
        #
        # print("J22VecCopy = {}".format(test))

        # theta
        thetaVec[el] = atan2(- np.dot(E2, matDir), np.dot(E1, matDir))

        # Rodrigues rotation matrix
        R[0, 0] = cos(matDirAngle) + normalVecX[el] * normalVecX[el] * (1 - cos(matDirAngle))
        R[0, 1] = normalVecX[el] * normalVecY[el] * (1 - cos(matDirAngle)) - \
                  normalVecZ[el] * sin(matDirAngle)
        R[0, 2] = normalVecY[el] * sin(matDirAngle) + \
                  normalVecX[el] * normalVecZ[el] * (1 - cos(matDirAngle))

        R[1, 0] = normalVecZ[el] * sin(matDirAngle) + \
                  normalVecX[el] * normalVecY[el] * (1 - cos(matDirAngle))
        R[1, 1] = cos(matDirAngle) + normalVecY[el] * normalVecY[el] * (1 - cos(matDirAngle))
        R[1, 2] = normalVecY[el] * normalVecZ[el] * (1 - cos(matDirAngle)) - \
                  normalVecX[el] * sin(matDirAngle)

        R[2, 0] = normalVecX[el] * normalVecZ[el] * (1 - cos(matDirAngle)) - \
                  normalVecY[el] * sin(matDirAngle)
        R[2, 1] = normalVecX[el] * sin(matDirAngle) + \
                  normalVecY[el] * normalVecZ[el] * (1 - cos(matDirAngle))
        R[2, 2] = cos(matDirAngle) + normalVecZ[el] * normalVecZ[el] * (1 - cos(matDirAngle))

        # find ed material unit vector on cylinder
        ed = np.matmul(R, matDir)

        # local material angle
        thetaCylinder[el] = atan2(np.dot(E2, ed), np.dot(E1, ed))

    data.LCable = LCable

    if data.matDirType == "global":
        data.thetaVec = thetaVec
    elif data.matDirType == "specific":
        print("nothing there yet")
        assert(0)
    elif data.matDirType == "cylinder":
        # cylinder axis
        data.thetaVec = thetaCylinder
    else:
        raise Exception("Specify matDirType to either global or specific")

    data.J11Vec = J11Vec
    data.J12Vec = J12Vec
    data.J22Vec = J22Vec

    data.normalVecX = normalVecX
    data.normalVecY = normalVecY
    data.normalVecZ = normalVecZ

    data.nnodes = nnodes
    data.nelems = nelems
    data.ndof = data.dim * nnodes
    data.nelemsMem = nelemsMem
    data.nelemsCable = nelemsCable

    data.X0 = X0
    data.Y0 = Y0
    data.Z0 = Z0
    cdef double [:] X = np.zeros(data.nnodes, dtype=np.double)
    cdef double [:] Y = np.zeros(data.nnodes, dtype=np.double)
    cdef double [:] Z = np.zeros(data.nnodes, dtype=np.double)

    cdef double [:] u = np.zeros(data.ndof, dtype=np.double)
    cdef double [:] V = np.zeros(data.ndof, dtype=np.double)
    cdef double [:] acc = np.zeros(data.ndof, dtype=np.double)
    cdef unsigned int index

    if data.restart is True:
        index = 0
        if data.dim == 2:
            for i in range(data.nnodes):
                u[index] = u_restart[i]
                u[index+1] = v_restart[i]
                index += 2
        else:
            for i in range(data.nnodes):
                u[index] = u_restart[i]
                u[index+1] = v_restart[i]
                u[index+2] = w_restart[i]
                index += 3

        for i in range(data.nnodes):
            X[i] = X0[i] + u_restart[i]
            Y[i] = Y0[i] + v_restart[i]
            Z[i] = Z0[i] + w_restart[i]

    else:

        for i in range(data.nnodes):
            X[i] = X0[i]
            Y[i] = Y0[i]
            Z[i] = Z0[i]

    data.X = X
    data.Y = Y
    data.Z = Z

    data.u = u
    data.V = V
    data.acc = acc

    data.Nc = Nc
    data.Nm = Nm

    # data.prescribedDof = prescribedDof
    # data.prescribedDisplacement = prescribedDisplacement
    data.loadedBCNodes = loadedBCNodes
    data.loadedBCEdges = loadedBCEdges
    data.loadedBCSurface = loadedBCSurface
    data.elPressurised = elPressurised
    data.nPressurised = nPressurised
    data.elFSI = elFSI
    data.nFSI = nFSI
    data.p = p

    data.Sx = Sx
    data.Sy = Sy
    data.Sz = Sz
    data.pFSI = pFSI

    data.cc = cc

    # element properties
    cdef:
        double [:] h = np.zeros(nelemsMemTemp, dtype=np.double)
        double [:] area2 = np.zeros(nelemsCableTemp, dtype=np.double)
        double [:] rho2 = np.zeros(nelemsCableTemp, dtype=np.double)
        double [:] rho3 = np.zeros(nelemsMemTemp, dtype=np.double)
        double [:] YoungsMod2 = np.zeros(nelemsCableTemp, dtype=np.double)
        double [:] YoungsMod3 = np.zeros(nelemsMemTemp, dtype=np.double)
        double [:] PoissonsRat = np.zeros(nelemsMemTemp, dtype=np.double)
        double [:, ::1] Ew = np.zeros((nelemsMemTemp, 3), dtype=np.double)

    for i in range(nelemsCable):
        # TODO: either d or area definable
        # element ID
        el = Nc[i, 0]
        area2[i] = np.pi * data.elStruc[bnames[el]]["d"] ** 2 / 4
        YoungsMod2[i] = data.elStruc[bnames[el]]["E"]
        rho2[i] = data.elStruc[bnames[el]]["density"]

    for i in range(nelemsMem):

        # element ID
        el = Nm[i, 0]
        h[i] = data.elStruc[bnames[el]]["h"]
        YoungsMod3[i] = data.elStruc[bnames[el]]["E"]
        rho3[i] = data.elStruc[bnames[el]]["density"]
        PoissonsRat[i] = data.elStruc[bnames[el]]["nu"]

    # rod bending hinges for B sets
    buildRodHinges(data, Nc, nelemsCable, LCable, YoungsMod2, bnames)

    data.t = h
    data.area2 = area2
    data.rho3 = rho3
    data.rho2 = rho2
    data.E3 = YoungsMod3
    data.E2 = YoungsMod2
    data.nu = PoissonsRat

    data.setNames = bnames

    data.N1set = N1set
    data.N2set = N2set
    data.N3set = N3set

    data.Ew = Ew

    # wrinkling Jarasjarungkiat
    cdef double [:] P = np.ones(data.ndof, dtype=np.double)
    cdef double [:] alpha_array = np.zeros(data.ndof, dtype=np.double)

    data.P = P
    data.alpha_array = alpha_array

    # force_vector
    cdef double [:] force_vector = np.zeros(data.ndof, dtype=np.double)
    data.force_vector = force_vector

    # prestress in cable elements
    cdef double [:] pre_stress_cable = np.zeros(data.nelemsCable, dtype=np.double)
    cdef unsigned int [:] pre_active = np.zeros(4, dtype=np.uintc)

    if pre_stress_cable_ID:

        index = 0

        for i in pre_stress_cable_ID:

            ind = np.where(np.asarray(Nc)[:, 0] == i)[0]

            for j in range(len(ind)):
                pre_stress_cable[ind[j]] = pre_stress_cable_mag[index]

            index += 1

        pre_active[0] = 1

    data.pre_stress_cable = pre_stress_cable

    # prestress in cable elements
    cdef double [:] pre_strain_cable = np.zeros(data.nelemsCable, dtype=np.double)

    if pre_strain_cable_ID:

        index = 0

        for i in pre_strain_cable_ID:

            ind = np.where(np.asarray(Nc)[:, 0] == i)[0]

            for j in range(len(ind)):
                pre_strain_cable[ind[j]] = pre_strain_cable_mag[index]

            index += 1

        pre_active[1] = 1

    data.pre_strain_cable = pre_strain_cable

    # prestress in membrane elements
    cdef double [:, ::1] pre_stress_membrane = np.zeros((data.nelemsMem, 3), dtype=np.double)

    if pre_stress_membrane_ID:

        index = 0

        pre_stress_membrane_mag = [item for sublist in pre_stress_membrane_mag for item in sublist]

        for i in pre_stress_membrane_ID:

            ind = np.where(np.asarray(Nm)[:, 0] == i)[0]

            for j in range(len(ind)):
                pre_stress_membrane[ind[j], 0] = pre_stress_membrane_mag[3 * (index + 1) - 3]
                pre_stress_membrane[ind[j], 1] = pre_stress_membrane_mag[3 * (index + 1) - 2]
                pre_stress_membrane[ind[j], 2] = pre_stress_membrane_mag[3 * (index + 1) - 1]

            index += 1

        pre_active[2] = 1

    data.pre_stress_membrane = pre_stress_membrane

    # prestress in membrane elements
    cdef double [:, ::1] pre_strain_membrane = np.zeros((data.nelemsMem, 3), dtype=np.double)

    if pre_strain_membrane_ID:

        index = 0

        pre_strain_membrane_mag = [item for sublist in pre_strain_membrane_mag for item in sublist]

        for i in pre_strain_membrane_ID:

            ind = np.where(np.asarray(Nm)[:, 0] == i)[0]

            for j in range(len(ind)):
                pre_strain_membrane[ind[j], 0] = pre_strain_membrane_mag[3 * (index + 1) - 3]
                pre_strain_membrane[ind[j], 1] = pre_strain_membrane_mag[3 * (index + 1) - 2]
                pre_strain_membrane[ind[j], 2] = pre_strain_membrane_mag[3 * (index + 1) - 1]

            index += 1

        pre_active[3] = 1

    data.pre_strain_membrane = pre_strain_membrane
    data.pre_active = pre_active

    cdef double [:] pre_u = np.zeros(data.ndof, dtype=np.double)
    cdef double [:] M = np.zeros(data.ndof, dtype=np.double)

    pre_u_read = np.asarray(pre_u_read)
    pre_v_read = np.asarray(pre_v_read)

    data.dof_homogeneous = dofFixed

    if data.dim == 2:

        fix_dof = []

        for i in range(np.size(pre_u_read, 0)):

            pre_u[2 * (int(pre_u_read[i, 0]) + 1) - 2] = pre_u_read[i, 1]

            fix_dof.append(2 * (int(pre_u_read[i, 0]) + 1) - 2)

        for i in range(np.size(pre_v_read, 0)):

            pre_u[2 * (int(pre_v_read[i, 0]) + 1) - 1] = pre_v_read[i, 1]

            fix_dof.append(2 * (int(pre_v_read[i, 0]) + 1) - 1)

        fix_dof = np.asarray(fix_dof, dtype=np.intc)

        dofFixed = np.unique(np.append(dofFixed, fix_dof))

    elif data.dim == 3 and (pre_u_read or pre_v_read):
        raise Exception("Prescribed displacements not implemented in 3D.")
        # for i in range(np.size(pre_u_read, 0)):
        #     pre_u[3 * (int(pre_u_read[i, 0]) + 1) - 3] = pre_u_read[i, 1]
        #     pre_u[3 * (int(pre_u_read[i, 0]) + 1) - 2] = pre_u_read[i, 2]
        #     pre_u[3 * (int(pre_u_read[i, 0]) + 1) - 1] = pre_u_read[i, 3]

    if np.size(pre_u_read, 0) == 0:
        pre_u_dof_temp = np.array([-1], dtype=np.intc)
    else:
        data.pre_u_flag = True
        pre_u_dof_temp = np.asarray(fix_dof, dtype=np.intc)

    cdef int [:] pre_u_dof = pre_u_dof_temp

    data.dofFixed = dofFixed
    data.n_homogeneous_BC = n_homogeneous_BC

    data.pre_u_dof = pre_u_dof
    data.pre_u = pre_u
    data.M = M


def buildRodHinges(data, int [:, ::1] Nc, unsigned int nelemsCable, double [:] LCable, double [:] E2, bnames):
    """
    Build the bending hinges of rods made of "B" elements.

    A "B" element is a "C" bar (axial, carries compression) plus bending. Every set of type "B"
    (EI = E pi d^4 / 64, or an explicit "EI") is split into rods (connected chains of its line elements). Each rod is straight and stress-free in
    its rest state, so a hinge is placed at every interior node (node shared by exactly two
    elements of the same set). Free rod ends get no hinge, and different sets never share a
    hinge. A node shared by more than two elements of the set is a branch: no hinge is placed
    there and a warning is printed.

    sets data.Nb [nBend, 3] (node 0, middle node, node 2), data.nBend, data.kBend (EI / l0),
    data.EIBend, data.l0Bend, data.rBend (outer radius d / 2), data.rodOfHinge and
    data.elOfHinge [nBend, 2] (the two cable elements meeting at the hinge)
    """
    hinges = []
    kB = []
    EIList = []
    l0List = []
    rList = []
    rodID = []
    elOfHinge = []
    nRods = 0
    summary = []

    NcArr = np.asarray(Nc)[:nelemsCable]
    L0Arr = np.asarray(LCable)[:nelemsCable]
    E2Arr = np.asarray(E2)[:nelemsCable]

    for pid in (sorted(set(NcArr[:, 0].tolist())) if nelemsCable > 0 else []):

        name = bnames[pid]
        props = data.elStruc[name]

        if props["type"] != "B":
            continue

        els = np.where(NcArr[:, 0] == pid)[0]

        if "EI" in props:
            EI = float(props["EI"])
        else:
            EI = float(E2Arr[els[0]]) * np.pi * props["d"] ** 4 / 64

        # node -> {neighbour node: element}
        adj = {}
        for e in els:
            a, b = int(NcArr[e, 1]), int(NcArr[e, 2])
            if a == b:
                continue
            adj.setdefault(a, {})[b] = e
            adj.setdefault(b, {})[a] = e

        # connected components = rods
        comp = {}
        nRodsSet = 0
        for start in adj:
            if start in comp:
                continue
            stack = [start]
            while stack:
                n = stack.pop()
                if n in comp:
                    continue
                comp[n] = nRods + nRodsSet
                stack.extend(m for m in adj[n] if m not in comp)
            nRodsSet += 1

        nBranch = 0
        nHingeSet = 0
        phiMax = 0.
        kappaMax = 0.
        for n, nb in adj.items():
            if len(nb) > 2:
                nBranch += 1
                continue
            if len(nb) != 2:
                continue
            (n0, e0), (n2, e1) = nb.items()
            l0 = 0.5 * (L0Arr[e0] + L0Arr[e1])
            hinges.append([n0, n, n2])
            kB.append(EI / l0)
            EIList.append(EI)
            l0List.append(l0)
            rList.append(props["d"] / 2 if "d" in props else 0.)
            rodID.append(comp[n])
            elOfHinge.append([e0, e1])
            nHingeSet += 1

            # initial turning angle (rest state is straight)
            p0 = np.array([data.X0[n0], data.Y0[n0], data.Z0[n0]])
            p1 = np.array([data.X0[n], data.Y0[n], data.Z0[n]])
            p2 = np.array([data.X0[n2], data.Y0[n2], data.Z0[n2]])
            ea = p1 - p0
            eb = p2 - p1
            c = np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb))
            phi = np.arccos(np.clip(c, -1., 1.))
            phiMax = max(phiMax, phi)
            kappaMax = max(kappaMax, 2 * np.tan(phi / 2) / l0)

        if nBranch > 0:
            print("WARNING: set {}: {} node(s) shared by more than two elements, "
                  "no bending hinge placed there".format(name, nBranch))

        summary.append((name, nRodsSet, nHingeSet, EI, phiMax, kappaMax,
                        props.get("d", 0.) / 2 * float(E2Arr[els[0]]) * kappaMax))
        nRods += nRodsSet

    nBend = len(hinges)

    if nBend == 0:
        data.Nb = np.zeros((1, 3), dtype=np.intc)
        data.kBend = np.zeros(1, dtype=np.double)
    else:
        data.Nb = np.ascontiguousarray(hinges, dtype=np.intc)
        data.kBend = np.asarray(kB, dtype=np.double)

    data.nBend = nBend
    data.EIBend = np.asarray(EIList, dtype=np.double)
    data.l0Bend = np.asarray(l0List, dtype=np.double)
    data.rBend = np.asarray(rList, dtype=np.double)
    data.rodOfHinge = np.asarray(rodID, dtype=np.intc)
    data.elOfHinge = np.asarray(elOfHinge, dtype=np.intc).reshape(-1, 2)
    data.nRods = nRods

    if data.silent is False:
        for name, nr, nh, EI, phiMax, kappaMax, sigma in summary:
            print("Rod bending in set {}: {} rod(s), {} hinge(s), EI = {:.3E} N m2, "
                  "initial max turning angle {:.1f} deg, max curvature {:.1f} 1/m "
                  "(R = {:.4f} m), max bending stress {:.3E} Pa".format(
                      name, nr, nh, EI, np.degrees(phiMax), kappaMax,
                      1 / kappaMax if kappaMax > 0 else np.inf, sigma))
