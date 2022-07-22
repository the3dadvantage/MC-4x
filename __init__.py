bl_info = {
    "name": "MC_main",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Extended Tools > Modeling Cloth",
    "description": "It's like cloth but in a computer!",
    "warning": "3D models of face masks will not protect your computer from viruses",
    "wiki_url": "",
    "category": '3D View'}

if "bpy" in locals():
    import imp
    imp.reload(MC_self_collision)
    imp.reload(MC_object_collision)
    imp.reload(MC_pierce)
    imp.reload(MC_flood)
    imp.relaod(MC_edge_collide)
    imp.reload(MC_grid)
    imp.reload(MC_main)
    #imp.reload(ModelingCloth)
    #imp.reload(SurfaceFollow)
    #imp.reload(UVShape)
    #imp.reload(DynamicTensionMap)
    print("Reloaded Modeling Cloth")
else:
    from . import MC_self_collision
    from . import MC_object_collision
    from . import MC_pierce
    from . import MC_flood
    from . import MC_edge_collide
    from . import MC_grid
    from . import MC_main
    print("Imported Modeling Cloth")

   
def register():
    #MC_self_collision.register()
    #MC_object_collision.register()
    #MC_pierce.register()
    MC_main.register()
    #ModelingCloth.register()    
    #SurfaceFollow.register()
    #UVShape.register()
    #DynamicTensionMap.register()

    
def unregister():
    #MC_self_collision.unregister()
    #MC_object_collision.unregister()
    #MC_pierce.unregister()
    MC_main.unregister()
    #ModelingCloth.unregister()
    #SurfaceFollow.unregister()
    #UVShape.unregister()
    #DynamicTensionMap.unregister()

    
if __name__ == "__main__":
    register()
