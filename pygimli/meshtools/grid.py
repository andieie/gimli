# -*- coding: utf-8 -*-
"""General grid generation and maintenance."""

import os

import numpy as np

import pygimli as pg

from .polytools import polyCreateWorld, syscallTetgen


def createGrid(x=None, y=None, z=None, **kwargs):
    """Create grid style mesh.

    Generate simple grid with defined node positions for each dimension.
    The resulting grid depends on the amount of given coordinate arguments
    and consists out of edges (1D - x), quads (2D- x and y), or
    hexahedrons(3D- x, y, and z).

    Parameters
    ----------
    kwargs:
        x: array
            x-coordinates for all Nodes (1D, 2D, 3D)
        y: array
            y-coordinates for all Nodes (2D, 3D)
        z: array
            z-coordinates for all Nodes (3D)
        marker : int = 0
            Marker for resulting cells.
        worldBoundaryMarker : bool = False
            Boundaries are enumerated with world marker, i.e., Top = -1
            All remaining = -2.
            Default marker are left=1, right=2, top=3, bottom=4, front=5, back=6

    Examples
    --------
    >>> import pygimli as pg
    >>> mesh = pg.meshtools.createGrid(x=[0, 1, 1.5, 2],
    ...                                y=[-1, -0.5, -0.25, 0], marker=2)
    >>> print(mesh)
    Mesh: Nodes: 16 Cells: 9 Boundaries: 24
    >>> _ = pg.show(mesh)
    >>> pg.wait()
    """
    if 'degree' in kwargs:
        return createGridPieShaped(x, **kwargs)

    if x is not None:
        if isinstance(x, int):
            x = list(range(x))
        kwargs['x'] = x
    if y is not None:
        if isinstance(y, int):
            y = list(range(y))
        kwargs['y'] = y
    if z is not None:
        if isinstance(z, int):
            z = list(range(z))
        kwargs['z'] = z

    return pg.core._pygimli_.createGrid(**kwargs)


def createGridPieShaped(x, degree=10.0, h=2, marker=0):
    """Create a 2D pie shaped grid (segment from annulus or cirlce).

    TODO:
    ----
    * degree: > 90 .. 360

    Arguments
    ---------
    x: array
        x-coordinates for all Nodes (2D). If you need it 3D, you can apply :py:mod:`pygimli.meshtools.extrudeMesh` on it.

    degree: float [None]
        Create a pie shaped grid for a value between 0 and 90.
        Creates an optional inner boundary (marker=2) for a annulus with x[0] > 0. Outer boundary marker is 1. Optional h refinement. Center node is the first for circle segment.

    h: int [2]
        H-Refinement for degree option.

    marker: int = 0
        Marker for resulting cells.

    Returns
    -------
    mesh: :gimliapi:`GIMLI::Mesh`

    Examples
    --------
    >>> import pygimli as pg
    >>> mesh = pg.meshtools.createGridPieShaped(x=[0, 1, 3], degree=45, h=3)
    >>> print(mesh)
    Mesh: Nodes: 117 Cells: 128 Boundaries: 244
    >>> _ = pg.show(mesh)
    >>> mesh = pg.meshtools.createGridPieShaped(x=[1, 2, 3], degree=45, h=3)
    >>> print(mesh)
    Mesh: Nodes: 153 Cells: 128 Boundaries: 280
    >>> _ = pg.show(mesh)
    """
    mesh = pg.Mesh(dim=2)

    for i in range(0, len(x)):
        mesh.createNodeWithCheck([x[i], 0.0])

        mesh.createNodeWithCheck([x[i]*np.cos(degree*np.pi/180),
                                  x[i]*np.sin(degree*np.pi/180)])

    if abs(x[0]) < 1e-6:
        mesh.createCell([0, 1, 2])
        for i in range(0, (len(x)-2)*2-1, 2):
            c = mesh.createCell([i+1, i+3, i+4, i+2])
    else:
        for i in range(0, len(x)*2-2, 2):
            c = mesh.createCell([i, i+2, i+3, i+1])
        mesh.createBoundary([0, 1], marker=1)

    mesh.createBoundary([mesh.nodeCount()-2, mesh.nodeCount()-1], marker=2)

    for i in range(h):
        mesh = mesh.createH2()
    mesh.createNeighbourInfos()

    for b in mesh.boundaries():
        if b.outside() and b.marker() == 0:
            if b.norm()[1] == 0.0:
                b.setMarker(4) # bottom
            else:
                b.setMarker(3)

    meshR = pg.Mesh(mesh)

    ## move all nodes on the inner boundary to rw
    for b in mesh.boundaries():
        line = pg.Line(b.node(0).pos(), b.node(1).pos())

        rSoll = line.intersect([0.0, 0.0], [1.0, 0.0])[0]
        if rSoll > 1e-4:
            for n in b.nodes():
                scale = rSoll/n.pos().abs()
                if scale > 1:
                    meshR.node(n.id()).setPos(pg.Line([0.0, 0.0], n.pos()).at(scale))

    if marker != 0:
        for c in meshR.cells():
            c.setMarker(marker)

    return meshR


def appendTriangleBoundary(mesh, xbound=10, ybound=10, marker=1, quality=34.0,
                           area=0.0, smooth=False, markerBoundary=1,
                           isSubSurface=False, verbose=False):
    """Add a triangle mesh boundary to a given mesh.

    Returns a new mesh that contains a triangulated box around a given mesh
    suitable for geo-simulation (surface boundary with marker = -1  at top and marker = -2 in the inner subsurface).
    The old boundary marker from mesh will be preserved, except for marker == -2 which will be switched to 2 since we assume
    -2 is the world marker for outer boundaries in the subsurface.

    Parameters
    ----------
    mesh : mesh object
        Mesh to which the triangle boundary should be appended.
    xbound : float, optional
        Horizontal prolongation distance. Minimal mesh 0.5 x extension.
    ybound : float, optional
        Vertical prolongation distance. Minimal mesh 0.5 y extension.
    marker : int, optional
        Marker of new cells.
    markerBoundary : int, optional
        Marker of the inner boundary edges between mesh and new boundary.
    quality : float, optional
        Triangle quality.
    area: float, optional
        Triangle max size within the boundary.
    smooth : boolean, optional
        Apply mesh smoothing.
    isSubSurface : boolean, optional
        Apply boundary conditions suitable for geo-simulation and prolongate
        mesh to the surface if necessary.
    verbose : boolean, optional
        Be verbose.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import pygimli as pg
    >>> from pygimli.viewer.mpl import drawMesh, drawModel
    >>> from pygimli.meshtools import appendTriangleBoundary
    >>> inner = pg.createGrid(range(5), range(5), marker=1)
    >>> mesh = appendTriangleBoundary(inner, xbound=3, ybound=6, marker=2)

    >>> fig, (ax1, ax2) = plt.subplots(1,2)
    >>> p1 = drawMesh(ax1, inner)
    >>> p2 = drawModel(ax2, mesh, mesh.cellMarkers(), label='marker')
    >>> p3 = drawMesh(ax2, mesh)
    >>> txt1 = ax1.set_title("a) Input grid")
    >>> txt2 = ax2.set_title("b) With triangle boundary")

    See Also
    --------
    appendTetrahedronBoundary
    """
    surface = 0.0

    # find boundaries on left/right/bottom/top side
    le = [b for b in mesh.boundaries() if b.center().x() == mesh.xmin()]
    bo = [b for b in mesh.boundaries() if b.center().y() == mesh.ymin()]
    ri = [b for b in mesh.boundaries() if b.center().x() == mesh.xmax()]
    top = [b for b in mesh.boundaries() if b.center().y() == mesh.ymax()]

    # gather all right boundary nodes after sorting in boundaryNodes
    tmp = []
    for b in ri:
        if b.node(0) not in tmp:
            tmp.append(b.node(0))
        if b.node(1) not in tmp:
            tmp.append(b.node(1))

    tmp.sort(key=lambda n: n.pos().y())
    tmp.reverse()
    boundaryNodes = tmp

    # gather all bottom boundary nodes and add them to boundaryNodes
    boNode = []
    for b in bo:
        if b.node(0) not in boNode + boundaryNodes:
            boNode.append(b.node(0))
        if b.node(1) not in boNode + boundaryNodes:
            boNode.append(b.node(1))

    boNode.sort(key=lambda n: n.pos().x())
    boNode.reverse()
    boundaryNodes = boundaryNodes + boNode

    # gather all left boundary nodes and add them to boundaryNodes
    tmp = []
    for b in le:
        if b.node(0) not in tmp + boundaryNodes:
            tmp.append(b.node(0))
        if b.node(1) not in tmp + boundaryNodes:
            tmp.append(b.node(1))

    tmp.sort(key=lambda n: n.pos().y())
    boundaryNodes = boundaryNodes + tmp

    if isSubSurface:
        # gather all top boundary nodes and add them to boundaryNodes
        topNodes = []
        for boundary in top:
            if boundary.node(0) not in topNodes + boundaryNodes:
                topNodes.append(boundary.node(0))
            if boundary.node(1) not in topNodes + boundaryNodes:
                topNodes.append(boundary.node(1))
        topNodes.sort(key=lambda n: n.pos().x())
        boundaryNodes = boundaryNodes + topNodes

    poly = pg.Mesh()

    preserveSwitch = ''

    if isSubSurface:
        # add all boundary nodes
        for n in boundaryNodes:
            poly.createNode(n.pos())

        # and connect them by a closed polygon
        for i in range(0, poly.nodeCount()):
            poly.createEdge(
                poly.node(i), poly.node((i + 1) % poly.nodeCount()),
                markerBoundary)

        # add four corners of the world box
        xtLen = 12
        # x bottom boundary sampling points
        # xBottom = pg.asvector(np.linspace(mesh.xmin() - xbound,
        #                                   mesh.xmax() + xbound, xtLen))

        n1 = poly.createNode(pg.RVector3(mesh.xmax() + xbound, surface, 0.0))
        n2 = poly.createNode(pg.RVector3(mesh.xmin() - xbound, surface, 0.0))
        n3 = poly.createNode(pg.RVector3(mesh.xmin() - xbound, mesh.ymin() -
                                         ybound, 0.0))
        n4 = poly.createNode(pg.RVector3(mesh.xmax() + xbound, mesh.ymin() -
                                         ybound, 0.0))

        # and connect them by a closed polygon
        poly.createEdge(n1, n2, pg.core.MARKER_BOUND_HOMOGEN_NEUMANN)
        poly.createEdge(n2, n3, pg.core.MARKER_BOUND_MIXED)
        poly.createEdge(n3, n4, pg.core.MARKER_BOUND_MIXED)
        poly.createEdge(n4, n1, pg.core.MARKER_BOUND_MIXED)

    else:  # no isSubSurface
        xbound = max(xbound, 0.5 * (mesh.xmax() - mesh.xmin()))
        ybound = max(ybound, 0.5 * (mesh.ymax() - mesh.ymin()))
        # add top right node and boundary nodes

        dxMin = boNode[0].pos().distance(boNode[1].pos()) * 1.1
        xtLen = max(5, int(xbound / dxMin / 2.))

        # x top boundary sampling points
        xTop = pg.core.increasingRange(dxMin, xbound, xtLen)
        # y boundary sampling points
        yLeft = pg.core.increasingRange(xTop[len(xTop)-1] - xTop[len(xTop)-2],
                                        abs(mesh.ymin() - ybound), xtLen)

        xtLen = max(5, int((mesh.xmax() - mesh.xmin()) / dxMin / 2.))

        # x bottom boundary sampling points
        xBottom = pg.Vector(np.linspace(mesh.xmin() - xbound,
                                        mesh.xmax() + xbound, 2 * xtLen))

        for i, val in enumerate(pg.core.fliplr(xTop)[0:len(xTop) - 1]):
            poly.createNode([mesh.xmax() + val, mesh.ymax(), 0.0])

        for n in boundaryNodes:
            poly.createNode(n.pos())

        # add top left, bottom left and bottom right node

        for t in xTop[1:len(xTop)]:
            poly.createNode([mesh.xmin() - t, mesh.ymax(), 0.0])

        for t in yLeft[1:len(yLeft)]:
            poly.createNode([mesh.xmin() - xbound, mesh.ymax() - t, 0.0])

        for t in xBottom[1:len(xBottom) - 1]:
            poly.createNode([t, mesh.ymin() - ybound, 0.0])

        for t in pg.core.fliplr(yLeft)[0:len(yLeft) - 1]:
            poly.createNode([mesh.xmax() + xbound, mesh.ymax() - t, 0.0])

        # create a closed polygon through all new nodes
        for i in range(0, poly.nodeCount()):
            poly.createEdge(
                poly.node(i), poly.node((i + 1) % poly.nodeCount()),
                markerBoundary)

        preserveSwitch = 'Y'
    # poly.exportVTK('out.poly')

    mesh2 = pg.Mesh(2)

    # call triangle mesh generation
    triswitches = '-pzeAfa' + preserveSwitch + 'q' + str(quality)

    if area > 0:
        triswitches += 'a' + str(area)

    if not verbose:
        triswitches += 'Q'

    if isSubSurface:
        margin = 0.0001
        poly.addHoleMarker(pg.RVector3(mesh.xmin() + margin, mesh.ymax() -
                                       margin))
        tri = pg.core.TriangleWrapper(poly)
        tri.setSwitches(triswitches)
        tri.generate(mesh2)
    else:
        pg.core.TriangleWrapper(poly, mesh2, triswitches)

    if smooth:
        mesh2.smooth(nodeMoving=True, edgeSwapping=True, smoothFunction=1,
                     smoothIteration=2)

    mesh2.setCellMarkers([marker] * mesh2.cellCount())

    # map does copies the cell not the reference, this should not happen **TODO check 20210305
    # map(lambda cell: mesh2.copyCell(cell), mesh2.cells())
    for c in mesh.cells():
        mesh2.copyCell(c)
    
    # we need to delete the old boundary markers or the new neighbour infos will fail for old outside boundaries
    mesh2.setBoundaryMarkers(np.zeros(mesh2.boundaryCount()))
    mesh2.createNeighborInfos(force=True)

    for b in mesh.boundaries():
        if b.marker() != 0:
            b2 = mesh2.copyBoundary(b)
            
            # some automagic .. original mesh contains bmarker == -2 which means mixed condition
            # this special marker will be switched to 2
            if b.marker() == -2:
                b2.setMarker(2)
                
    for b in mesh2.boundaries():
        if b.outside():
            if b.norm().x() != 0 or b.norm().y() == -1.0:
                b.setMarker(pg.core.MARKER_BOUND_MIXED)
            else:
                b.setMarker(pg.core.MARKER_BOUND_HOMOGEN_NEUMANN)

    return mesh2


def appendTetrahedronBoundary(mesh, xbound=100, ybound=100, zbound=100,
                              marker=1, quality=2, isSubSurface=False,
                              verbose=False):
    """    Return new mesh surrounded by tetrahedron boundary box.

    Creates a tetrahedral box around a given mesh
    suitable for geo-simulation (surface boundary at top).

    Parameters
    ----------
    mesh : mesh object
        Mesh to which the tetrahedron boundary should be appended.
    xbound : float, optional
        Horizontal prolongation distance in x-direction.
    ybound : float, optional
        Horizonal prolongation distance in y-direction.
    zbound : float, optional
        Vertical prolongation distance.
    marker : int, optional
        Marker of new cells.
    quality : float, optional
        Triangle quality.
    isSubSurface : boolean, optional
        Apply boundary conditions suitable for geo-simulaion and prolongate
        mesh to the surface if necessary.
    verbose : boolean, optional
        Be verbose.

    See Also
    --------
    appendTriangleBoundary

    Notes
    -----
    Boundaries of mesh need marker 1.
    """
    DeprecationWarning()
    # create boundary for mesh from boundary marker == 1

    if isSubSurface:
        raise Exception('Implement me')

    meshBoundary = pg.Mesh()
    meshBoundary.createH2()

    bounds = []
    for b in meshBoundary.boundaries():
        if b.marker() == 1:
            bounds.append(b)

    meshBoundaryPoly = pg.Mesh()
    meshBoundaryPoly.createMeshByBoundaries(
        meshBoundary, meshBoundary.findBoundaryByMarker(1))

    meshBoundaryPoly.exportAsTetgenPolyFile("paraBoundary.poly")
    # system( 'polyConvert -V paraBoundary' )

    # create worldSurface.poly including boundary mesh for a nice surface mesh
    # it will be later the tetgen input with preserve boundary
    polyCreateWorld('worldSurface', x=xbound, y=ybound, depth=zbound, marker=1,
                    verbose=verbose)
    os.system('polyMerge -N worldSurface paraBoundary worldSurface')
    # There syscalls are not working!
    polyAddVIP('worldSurface', mesh.cell(0).center(), isHoleMarker=True,
               verbose=verbose)
    worldBoundary = tetgen('worldSurface', quality=1.12, verbose=verbose)
    # worldBoundary.exportBoundaryVTU('worldSurface')

    worldPoly = pg.Mesh()
    worldPoly.createMeshByBoundaries(worldBoundary,
                                     worldBoundary.findBoundaryByMarker(-2, 0))
    worldPoly.exportAsTetgenPolyFile("worldSurface.poly")

    os.system('polyMerge -N worldSurface paraBoundary boundaryWorld')

    # mesh should have to be a hole
    polyAddVIP('boundaryWorld', mesh.cell(0).center(), isHoleMarker=True,
               verbose=verbose)
    # system( 'polyConvert -o world-poly -V boundaryWorld' )

    boundMesh = tetgen('boundaryWorld', quality=quality, preserveBoundary=True,
                       verbose=verbose)
    # boundMesh.exportVTK( 'boundaryWorld' )

    # merge mesh and worldBoundary
    for c in boundMesh.cells():
        c.setMarker(marker)

    if verbose:
        print("merge grid and boundary")

    swatch = pg.core.Stopwatch(True)
    for c in meshBoundary.cells():
        nodes = pg.stdVectorNodes()
        for n in c.nodes():
            nodes.append(boundMesh.createNodeWithCheck(n.pos()))

        boundMesh.createCell(nodes, c.marker())
    if verbose:
        print(" done.", swatch.duration(True))

    try:
        os.remove('boundaryWorld.bms')
        os.remove('worldSurface.bms')
        os.remove('boundaryWorld.poly')
        os.remove('paraBoundary.poly')
        os.remove('worldSurface.poly')
    except BaseException as e:
        print(e)

    return boundMesh


if __name__ == "__main__":
    pass
