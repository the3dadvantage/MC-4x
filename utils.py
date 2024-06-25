import bpy
import bmesh
import numpy as np
import mathutils as mu


LONG = True
LONG = False
if LONG:
    D_TYPE_F = np.float64
    D_TYPE_I = np.int64
else:
    D_TYPE_F = np.float32
    D_TYPE_I = np.int32

#=======================#
# WEB ------------------#
#=======================#
# web
def open_browser(link=None):
    import webbrowser

    if link == "paypal":
        # subscribe with paypal:
        link = "https://www.paypal.com/webapps/billing/plans/subscribe?plan_id=P-8V2845643T4460310MYYRCZA"
    if link == "gumroad":    
        # subscribe with gumroad:
        link = "https://richcolburn.gumroad.com/l/wooww"
    if link == "patreon":    
        # subscribe with patreon:
        link = "https://www.patreon.com/checkout/TheologicalDarkWeb?rid=23272750"
    if link == "donate":    
        # paypal donate:
        link = "https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=4T4WNFQXGS99A"
    
    webbrowser.open(link)


#=======================#
# DEBUG ----------------#
#=======================#
# debug
def popup_error(msg):
    def oops(self, context):
        return
    bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')


# python
def r_print(r, m=''):
    """Prints numpy arrays
    and lists rounded off."""
    print()
    print(m + " --------")
    if isinstance(r, list):
        for v in r:
            print(np.round(v, 3), m)
        return        
    print(np.round(r, 3), m)


#=======================#
# VERTEX GROUPS --------#
#=======================#
# vertex groups
def assign_vert_group(ob, v_list, name, weight=1.0):
    """Does what you might imagine
    (makes me a sandwich)."""
    if name not in ob.vertex_groups:
        nvg = ob.vertex_groups.new(name=name)
    else:
        nvg = ob.vertex_groups[name]
    nvg.add(v_list, weight, 'REPLACE')


# vertex groups
def get_weight(ob, v, group):
    """Try to get single weight. If vert not in group
    retun 0.0"""
    vertex_group = ob.vertex_groups[group]
    try:    
        weight = vertex_group.weight(v)
    except:
        return 0.0
    return weight
    

# vertex groups
def get_weights(ob, verts, group):
    """Try to get list of weights. If vert not in group
    set that vert to 0.0"""
    vertex_group = ob.vertex_groups[group]
    weights = np.zeros(len(verts), dtype=np.float32)
    for e, v in enumerate(verts):
        try:    
            weights[e] = vertex_group.weight(v)
        except:
            weights[e] = 0.0
    return weights


# vertex groups
def get_vertex_weights_overwrite(ob, group_name, default=1.0):
    """Get weights of the group. if it's not
    in the group set the weight of that
    vertex to default"""
    if group_name not in ob.vertex_groups:
        ob.vertex_groups.new(name=group_name)

    vertex_group = ob.vertex_groups[group_name]
    all_vertices = range(len(ob.data.vertices))
    weights = []

    for idx in all_vertices:
        try:
            weight = vertex_group.weight(idx)
            weights.append(weight)
        except RuntimeError:
            weights.append(default)

    vertex_weights = np.array(weights, dtype=np.float32)
    return vertex_weights


# vertex groups
def vertex_groups_to_dict(ob, verts=None, offset=None):
    """Create a dictionary of groups for every vert
    or a list of specific verts if "vert" is
    an array or list of numbers.
    "offset" is for breaking verts out into a new object"""

    vgroup_names = {vgroup.index: vgroup.name for vgroup in ob.vertex_groups}

    if verts is None:
        vgroups = {v.index: [vgroup_names[g.group] for g in v.groups] for v in ob.data.vertices}
        return vgroups

    if offset is None:
        vgroups = {
            ob.data.vertices[v].index: [vgroup_names[g.group] for g in ob.data.vertices[v].groups] for v in verts
        }
        return vgroups

    vgroups = {
        ob.data.vertices[v].index - offset[v]: [vgroup_names[g.group] for g in ob.data.vertices[v].groups]
        for v in verts
    }
    return vgroups


#=======================#
# BMESH ----------------#
#=======================#
# bmesh
def get_bmesh(ob=None, refresh=False, mesh=None, copy=False):
    """gets bmesh in editmode or object mode
    by checking the mode"""
    if ob.data.is_editmode:
        obm = bmesh.from_edit_mesh(ob.data)
        return obm
    obm = bmesh.new()
    m = ob.data
    if mesh is not None:
        m = mesh

    obm.from_mesh(m)
    if refresh:
        obm.verts.ensure_lookup_table()
        obm.edges.ensure_lookup_table()
        obm.faces.ensure_lookup_table()

    if copy:
        return obm.copy()

    return obm


# bmesh
def nearest_points_on_bmesh(coords, obm, return_all=False):
    """Uses blender closest_point... with
    a bmesh."""
    
    bvh = mu.bvhtree.BVHTree.FromBMesh(obm)
            
    faces = []
    locs = []
    norms = []
    dists = []
    
    for co in coords:
        loc, norm, face_index, dist = bvh.find_nearest(co)
        faces += [face_index]
        locs += [loc]
        if return_all:    
            norms += [norm]
            dists += [dist]
    if return_all:
        return faces, locs, norms, dists

    return faces, locs


# bmesh
def get_tridex(ob, teidx=False, tmesh=False, tobm=None, refresh=False, cull_verts=None, free=False):
    """Return an index for viewing the
    verts as triangles using a mesh and
    foreach_get.
    Return tridex and triobm"""

    if tobm is None:
        tobm = get_bmesh(ob)
        if cull_verts is not None:
            tobm.verts.ensure_lookup_table()
            cv = [tobm.verts[v] for v in cull_verts]
            bmesh.ops.delete(tobm, geom=cv, context='VERTS')

    bmesh.ops.triangulate(tobm, faces=tobm.faces)
    me = bpy.data.meshes.new('tris')
    tobm.to_mesh(me)
    p_count = len(me.polygons)
    tridex = np.empty((p_count, 3), dtype=D_TYPE_I)
    me.polygons.foreach_get('vertices', tridex.ravel())
    three_edges = np.empty((len(tobm.faces), 3), dtype=D_TYPE_I)
    for e, f in enumerate(tobm.faces):
        eid = [ed.index for ed in f.edges]
        three_edges[e] = eid

    if free:
        edge_normal_keys = []
        for e in tobm.edges:
            if len(e.link_faces) == 2:
                e_key = [e.link_faces[0].index, e.link_faces[1].index]
            elif len(e.link_faces) == 1:
                e_key = [e.link_faces[0].index, e.link_faces[0].index]
            else:
                e_key = [-1, -1]    
            edge_normal_keys += [e_key]
                
        eidx = np.array(me.edge_keys, dtype=D_TYPE_I)
        bpy.data.meshes.remove(me)
        tobm.free()
        return tridex, eidx, three_edges, np.array(edge_normal_keys, dtype=D_TYPE_I)
    
    if tmesh:
        return tridex, tobm, me
    if teidx:
        eidx = np.array(me.edge_keys, dtype=D_TYPE_I)
        # clear unused tri mesh
        bpy.data.meshes.remove(me)
        return tridex, eidx, tobm

    # clear unused tri mesh
    bpy.data.meshes.remove(me)

    if refresh:
        tobm.verts.ensure_lookup_table()
        tobm.edges.ensure_lookup_table()
        tobm.faces.ensure_lookup_table()

    return tridex, tobm, None


# bmesh
def get_tridex_2(ob, mesh=None): # faster than get_tridex()
    """Return an index for viewing the
    verts as triangles using a mesh and
    foreach_get. Faster than get_tridex()"""

    if mesh is not None:
        tobm = bmesh.new()
        tobm.from_mesh(mesh)
        bmesh.ops.triangulate(tobm, faces=tobm.faces)
        me = bpy.data.meshes.new('tris')
        tobm.to_mesh(me)
        p_count = len(me.polygons)
        tridex = np.empty((p_count, 3), dtype=np.int32)
        me.polygons.foreach_get('vertices', tridex.ravel())

        # clear unused tri mesh
        bpy.data.meshes.remove(me)
        if ob == 'p':
            return tridex, tobm

        tobm.free()
        return tridex

    if ob.data.is_editmode:
        ob.update_from_editmode()

    tobm = bmesh.new()
    tobm.from_mesh(ob.data)
    bmesh.ops.triangulate(tobm, faces=tobm.faces[:])
    me = bpy.data.meshes.new('tris')
    tobm.to_mesh(me)
    p_count = len(me.polygons)
    tridex = np.empty((p_count, 3), dtype=np.int32)
    me.polygons.foreach_get('vertices', tridex.ravel())

    # clear unused tri mesh
    bpy.data.meshes.remove(me)

    return tridex, tobm


#=======================#
# GEOMETRY -------------#
#=======================#
# geometry
def get_tridex_with_edges(ob, teidx=False, tmesh=False, tobm=None, refresh=False, cull_verts=None):
    """Return an index for viewing the
    verts as triangles using a mesh and
    foreach_get.
    Return tridex and triobm"""

    if tobm is None:
        tobm = get_bmesh(ob)
        if cull_verts is not None:
            tobm.verts.ensure_lookup_table()
            cv = [tobm.verts[v] for v in cull_verts]
            bmesh.ops.delete(tobm, geom=cv, context='VERTS')

    bmesh.ops.triangulate(tobm, faces=tobm.faces)
    me = bpy.data.meshes.new('tris')
    tobm.to_mesh(me)
    p_count = len(me.polygons)
    tridex = np.empty((p_count, 3), dtype=D_TYPE_I)
    me.polygons.foreach_get('vertices', tridex.ravel())
    if tmesh:
        return tridex, tobm, me
    if teidx:
        eidx = np.array(me.edge_keys, dtype=D_TYPE_I)
        # clear unused tri mesh
        bpy.data.meshes.remove(me)
        return tridex, eidx, tobm

    # clear unused tri mesh
    bpy.data.meshes.remove(me)

    if refresh:
        tobm.verts.ensure_lookup_table()
        tobm.edges.ensure_lookup_table()
        tobm.faces.ensure_lookup_table()

    return tridex, tobm, None


# geometry
def inside_triangles(tris, points, margin=0.0):  # , cross_vecs):
    """Checks if points are inside triangles"""
    origins = tris[:, 0]
    cross_vecs = tris[:, 1:] - origins[:, None]

    v2 = points - origins

    # ---------
    v0 = cross_vecs[:, 0]
    v1 = cross_vecs[:, 1]

    d00_d11 = np.einsum('ijk,ijk->ij', cross_vecs, cross_vecs)
    d00 = d00_d11[:, 0]
    d11 = d00_d11[:, 1]
    d01 = np.einsum('ij,ij->i', v0, v1)
    d02 = np.einsum('ij,ij->i', v0, v2)
    d12 = np.einsum('ij,ij->i', v1, v2)

    div = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * div
    v = (d00 * d12 - d01 * d02) * div

    w = 1 - (u + v)
    # !!!! needs some thought
    # margin = 0.0
    # !!!! ==================
    weights = np.array([w, u, v]).T
    # check = (u >= margin) & (v >= margin) & (w >= margin)
    check = (u >= margin) & (v >= margin) & (w >= margin)
    # check = ~np.any(weights > 1.0, axis=1)
    return check, weights


# geometry
def closest_point_edges(vecs, origins, p):
    '''Returns the location of the point on the edges'''
    vec2 = p - origins
    d = np.einsum('ij,ij->i', vecs, vec2) / np.einsum('ij,ij->i', vecs, vecs)
    cp = origins + vecs * d[:, None]
    return cp, d


# geometry
def np_closest_point_mesh(co, ob=None, t_data=None, return_weights=False):
    """Returns closest point on mesh
    and type v, f, or e depending
    on if the closest place was
    on a vert, a face, or an edge."""

    face = False
    edge = False
    edge_weight = None
    tri_weight = None

    # faces
    if t_data is None:
        tridex, teidx, tobm = get_tridex_with_edges(ob, teidx=True, tmesh=False)
    else:
        tridex, teidx, tobm = t_data['tridex'], t_data['teidx'], t_data['tobm']

    if ob is None:
        obco = np.array([v.co for v in tobm.verts], dtype=D_TYPE_F)
    else:
        obco = get_co(ob)

    tris = obco[tridex]
    check, weights = inside_triangles(tris, co, margin=0.0)
    checked_weights = weights[check]
    
    if np.any(check):
        face = True
        tidx = np.arange(tris.shape[0])[check]
        weight_plot = tris[check] * weights[check][:, :, None]
        locs = np.sum(weight_plot, axis=1)
        dif = co - locs
        dots = np.einsum('ij,ij->i', dif, dif)
        f_min = np.argmin(dots)
        f_loc = locs[f_min]
        tid = tidx[f_min]
        tri_weight = checked_weights[f_min]
        
    # edges
    teco = obco[teidx]
    vecs = teco[:, 1] - teco[:, 0]
    cpoes, dots = closest_point_edges(vecs, teco[:, 0], co)
    check = (dots > 0.0) & (dots < 1.0)
    
    if np.any(check):
        edge = True
        check_dots = dots[check]
        eidc = np.arange(check.shape[0])[check]
        checked = cpoes[check]
        dif = co - checked
        dots = np.einsum('ij,ij->i', dif, dif)
        emin = np.argmin(dots)
        e_loc = checked[emin]
        eid = eidc[emin]
        edge_weight = check_dots[emin]

    # verts
    dif = co - obco
    dots = np.einsum('ij,ij->i', dif, dif)
    vmin = np.argmin(dots)
    v_loc = obco[vmin]

    trifecta = [v_loc]
    type = ['v']
    ind = [vmin]
    tobm.verts.ensure_lookup_table()
    tobm.verts[vmin].normal

    if face:
        trifecta += [f_loc]
        type += ['f']
        ind += [tid]

    if edge:
        trifecta += [e_loc]
        type += ['e']
        ind += [eid]

    npl = np.array(trifecta, dtype=D_TYPE_F)
    dif = co - npl
    dots = np.einsum('ij,ij->i', dif, dif)
    amin = np.argmin(dots)
    loc = npl[amin]  # and all the chistians said "amin."
    t = type[amin]
    index = np.array(ind, dtype=D_TYPE_I)[amin]
    # location, type, index
    if return_weights:
        return loc, t, index, tri_weight, tridex, teidx, edge_weight
    return loc, t, index


# geometry
def closest_face_normal(co, t_data=None):
    """Manages v f and e types to get
    normal from np_closest_point_mesh()"""
    location, type, index = np_closest_point_mesh(co, t_data=t_data)
    obm_ex = t_data["tobm"]
    
    if type == 'f':
        return np.array(obm_ex.faces[index].normal, dtype=D_TYPE_F)
    if type == 'v':
        return np.array(obm_ex.verts[index].normal, dtype=D_TYPE_F)
    if type == 'e':
        eco = np.array([f.normal for f in obm_ex.edges[index].link_faces], dtype=D_TYPE_F)
        return np.mean(eco, axis=0)


# geometry
def u_vecs(vecs, dist=False):
    length = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
    u_v = vecs / length[:, None]
    if dist:
        return u_v, length
    return u_v


# geometry
def compare_vecs(v1, v2):
    return np.einsum('ij,ij->i', v1, v2)


# geometry
def measure_vecs(vecs, out=None, out2=None):
    return np.sqrt(np.einsum('ij,ij->i', vecs, vecs, out=out), out=out2)


#=======================#
# GET/SET --------------#
#=======================#
# universal ---------------
def get_proxy_co(ob, co=None, proxy=None, return_proxy=False):
    """Gets co with modifiers like cloth"""
    if proxy is None:

        dg = bpy.context.evaluated_depsgraph_get()
        prox = ob.evaluated_get(dg)
        proxy = prox.to_mesh()

    if co is None:
        vc = len(proxy.vertices)
        co = np.empty((vc, 3), dtype=np.float32)

    proxy.vertices.foreach_get('co', co.ravel())
    if return_proxy:
        return co, proxy, prox

    ob.to_mesh_clear()
    return co


# get/set
def absolute_co(ob, co=None):
    """Get proxy vert coords in world space"""
    co, proxy, prox = get_proxy_co(ob, co, return_proxy=True)
    m = np.array(ob.matrix_world, dtype=np.float32)
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc, proxy, prox
    
    
# get/set
def get_co(ob):
    """Returns Nx3 cooridnate set as numpy array"""
    co = np.empty((len(ob.data.vertices), 3), dtype=D_TYPE_F)
    ob.data.vertices.foreach_get("co", co.ravel())
    return co


# get/set
def get_shape_co(ob, shape='Key_1', co=None):
    if co is None:
        co = np.empty((len(ob.data.vertices), 3), dtype=D_TYPE_F)
    ob.data.shape_keys.key_blocks[shape].data.foreach_get('co', co.ravel())
    return co


# get/set
def local_co(ob, obs):
    """Currently using deformed co to return coords
    of objects in obs in the local space of ob."""
    OBCOS = []
    WMDS = []
    for sob in obs:
        OBCOS += [deform_co(sob)]
        WMDS += [sob.matrix_world] # why monsters don't sweat
    
    mesh_matrix = ob.matrix_world
    local_matrix = np.linalg.inv(mesh_matrix) @ np.array(WMDS, dtype=np.float32)
    
    local_co = np.empty((0, 3), dtype=np.float32)    
    for i in range(len(OBCOS)):
        ob_space = OBCOS[i] @ local_matrix[i][:3, :3].T
        ob_space += local_matrix[i][:3, 3]
        local_co = np.concatenate((local_co, ob_space), axis=0)
    
    return local_co


# get/set
def deform_co(ob, return_prox=False, triangulate=False, return_tobm=False):
    """Uses a list of mods that only deform,
    creates a proxy turning of every other
    mod then returns the proxy coords."""
    deform_mods = ['ARMATURE',
                   'CAST',
                   'CURVE',
                   'DISPLACE',
                   'HOOK',
                   'LAPLACIANDEFORM',
                   'LATTICE',
                   'MESH_DEFORM',
                   'SHRINKWRAP',
                   'SIMPLE_DEFORM',
                   'SMOOTH',
                   'CORRECTIVE_SMOOTH',
                   'LAPLACIANSMOOTH',
                   'SURFACE_DEFORM',
                   'WARP',
                   'WAVE',
                   'VOLUME_DISPLACE',
                   'CLOTH',
                   'SOFT_BODY',
                   'VERTEX_WEIGHT_EDIT',
                   'VERTEX_WEIGHT_MIX',
                   'VERTEX_WEIGHT_PROXIMITY',
                   ]        
    
    display = [m.show_viewport for m in ob.modifiers]
    for m in ob.modifiers:
        if m.type not in deform_mods:
            m.show_viewport = False
    
    prox = prox_object(ob)#, triangulate=True)
    
    if triangulate:
        tobm = get_bmesh(prox, refresh=True)
        bmesh.ops.triangulate(tobm, faces=tobm.faces)
    
    co = get_co(prox)
    
    for e, m in enumerate(ob.modifiers):
        m.show_viewport = display[e]
    
    if return_prox:
        return co, prox
    if return_tobm:
        return co, tobm
    
    return co


#=======================#
# BLENDER OBJECTS ------#
#=======================#
# blender objects
def prox_object(ob):
    """Returns object including modifier and
    shape key effects."""
    dg = bpy.context.evaluated_depsgraph_get()
    return ob.evaluated_get(dg)






# notes and such ----------

# list of mods other than deforming mods
('DATA_TRANSFER',
'MESH_CACHE', 
'MESH_SEQUENCE_CACHE',
'NORMAL_EDIT',
'WEIGHTED_NORMAL',
'UV_PROJECT',
'UV_WARP',
'GREASE_PENCIL_COLOR',
'GREASE_PENCIL_TINT',
'GREASE_PENCIL_OPACITY',
'ARRAY',
'BEVEL',
'BOOLEAN',
'BUILD',
'DECIMATE',
'EDGE_SPLIT', 
'NODES', 'MASK', 'MIRROR',
'MESH_TO_VOLUME',
'MULTIRES', 'REMESH', 
'SCREW', 'SKIN', 
'SOLIDIFY', 
'SUBSURF', 
'TRIANGULATE', 
'VOLUME_TO_MESH',
'WELD', 'WIREFRAME', 
'GREASE_PENCIL_SUBDIV', 
'GREASE_PENCIL_MIRROR', 

'GREASE_PENCIL_NOISE', 'GREASE_PENCIL_OFFSET', 'GREASE_PENCIL_SMOOTH', 'GREASE_PENCIL_THICKNESS',
'COLLISION', 'DYNAMIC_PAINT', 'EXPLODE', 'FLUID', 'OCEAN', 'PARTICLE_INSTANCE',
'PARTICLE_SYSTEM', 
'SURFACE')
    
