"""
Uses cgal-swig-bindings https://github.com/CGAL/cgal-swig-bindings.

"""
from CGAL import CGAL_Polygon_mesh_processing
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_modifier


def do_intersect_single_frame_cgal(sbj_vertices, sbj_faces,
                                   obj_vertices, obj_faces):
    sbj_verts_frame = sbj_vertices
    obj_verts_frame = obj_vertices

    poly_sbj = create_polyhedron_from_vertices_and_faces(sbj_verts_frame, sbj_faces)
    poly_obj = create_polyhedron_from_vertices_and_faces(obj_verts_frame, obj_faces)

    do_intersect = CGAL_Polygon_mesh_processing.do_intersect(poly_sbj, poly_obj)

    return do_intersect


def do_intersect_cgal(sbj_vertices, sbj_faces,
                      obj_vertices, obj_faces):
    n_frames = sbj_vertices.shape[0]

    any_intersect = False

    for frame in range(n_frames):
        sbj_verts_frame = sbj_vertices[frame]
        obj_verts_frame = obj_vertices[frame]

        poly_sbj = create_polyhedron_from_vertices_and_faces(sbj_verts_frame, sbj_faces)
        poly_obj = create_polyhedron_from_vertices_and_faces(obj_verts_frame, obj_faces)

        do_intersect = CGAL_Polygon_mesh_processing.do_intersect(poly_sbj, poly_obj)

        if do_intersect:
            any_intersect = True

            # out_intersect_mesh = Polyhedron_3()
            # intersection = CGAL_Polygon_mesh_processing.corefine_and_compute_intersection(poly_sbj, poly_obj, out_intersect_mesh)

    return any_intersect


def create_polyhedron_from_vertices_and_faces(vertices, faces, debug=False):
    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]

    m = Polyhedron_modifier()
    m.begin_surface(n_verts, n_faces)

    for vert_idx in range(n_verts):
        m.add_vertex(Point_3(float(vertices[vert_idx, 0]),
                             float(vertices[vert_idx, 1]),
                             float(vertices[vert_idx, 2])))

    for face_id in range(n_faces):
        m.begin_facet()
        m.add_vertex_to_facet(int(faces[face_id, 0]))
        m.add_vertex_to_facet(int(faces[face_id, 1]))
        m.add_vertex_to_facet(int(faces[face_id, 2]))
        m.end_facet()

    P = Polyhedron_3()
    P.delegate(m)
    if debug:
        print("(v,f,e) = ", P.size_of_vertices(), P.size_of_facets(), divmod(P.size_of_halfedges(), 2)[0])

    m.clear()

    assert P.is_valid()
    # assert not CGAL_Polygon_mesh_processing.does_self_intersect(P)

    return P
