import bpy
import bmesh
import numpy as np
import time

try:
    from mc_pro import utils as U

except ImportError:
    U = bpy.data.texts['utils.py'].as_module()


def copy_armature(ob, obs):
    """Copy armature mods and settings from
    ob1 to ob2. Return a dict with obs names
    and lists of new mods."""
    
    mods = {}
    for obj in obs:
        for mod in ob.modifiers:
            if mod.type == "ARMATURE":
                cmod = obj.modifiers.get(mod.name, None)
                if not cmod:
                    cmod = obj.modifiers.new(mod.name, mod.type)
                    mods[obj.name] = cmod.name
                # collect names of writable properties
                properties = [p.identifier for p in mod.bl_rna.properties
                              if not p.is_readonly]

                # copy those properties
                for prop in properties:
                    setattr(cmod, prop, getattr(mod, prop))
    return mods
    
    
def copy_weights():
    
    #T = time.time()
    msg = 'all clear'
    ob = bpy.context.object
    if ob:
        if ob.type == "MESH":
            sel = [sel_ob for sel_ob in bpy.context.selected_objects if ((sel_ob.type == "MESH") & (sel_ob != ob))]
            if len(sel) > 0:                    
                vgroups = U.vertex_groups_to_dict(ob, verts=None, offset=None)
                
                tridex, teidx, tobm = U.get_tridex_with_edges(ob, teidx=True, tmesh=False)
                if tridex.shape[0] == 0:
                    if np.random.randint(10) == 7:
                        U.popup_error("There is a really big spider crawling up the inside of your trouser leg.")
                    else:        
                        U.popup_error("Need some faces in the active object")
                    return    
            
                tco, deform_tobm = U.deform_co(ob, return_prox=False, triangulate=True, return_tobm=True)
                for sob in sel:
                    cloth_co = U.local_co(ob, [sob])
                    tidx, locs = U.nearest_points_on_bmesh(cloth_co, deform_tobm)
                    locs = np.array(locs, dtype=np.float32)
                    check, bvh_weights = U.inside_triangles(tco[tridex[tidx]], locs, margin=0.0)

                    for vert, idx in enumerate(tidx): # object vert is the same as enum
                        all_groups = set()
                        verts = tridex[idx]
                        
                        for e, v in enumerate(verts):

                            for g in vgroups[v]:
                                all_groups.add(g)
                    
                        listed_groups = list(all_groups)
                        group_weights = np.zeros(len(listed_groups), dtype=np.float32)
                        for e, g in enumerate(listed_groups):
                            W = sum(U.get_weights(ob, verts, g) * bvh_weights[vert])
                            group_weights[e] = W
                            
                        for we, gr in zip(group_weights, listed_groups):    
                            U.assign_vert_group(sob, [vert], gr, weight=we)

                mods = copy_armature(ob, sel)
                for obs in sel:
                    if obs.name in mods:
                        bpy.context.view_layer.objects.active = obs    
                        bpy.ops.object.modifier_move_to_index(modifier=mods[obs.name], index=0)
                
                
            else:
                msg = "No selected mesh objects"
        else:
            msg = "Active object is not a mesh."
    else:
        msg = "No active object."
        
