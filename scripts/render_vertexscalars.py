# # if you want to call the toolbox the old way with `blender -b -P demo_XXX.py`, then uncomment these two lines
import sys, os
sys.path.append("./blender_utils/")
sys.path.append("./src/")
from readOBJ import readOBJ
from blenderInit import blenderInit
from setMeshScalars import setMeshScalars
from invisibleGround import invisibleGround
from setCamera import setCamera
from setLight_sun import setLight_sun
from shadowThreshold import shadowThreshold
from renderImage import renderImage
from colorObj import colorObj
from setMat_VColor import setMat_VColor
import bpy
import os
import numpy as np

def render_mesh(output_name,
                vertex_scalars,
                filename,
                location,
                rotation,
                scale,
                cmin,
                cmax,
                output_dir = './renderings/images',
                imgRes_x = 1080, 
                imgRes_y = 1080, 
                numSamples = 200,
                exposure = 1.5,
                color_type = 'vertex',
                color_map = 'YlGnBu',
                camLocation = (3, 0, 2),
                lookAtLocation = (0,0,0.5),
                focalLength = 45, # (UI: click camera > Object Data > Focal Length)
                lightPosition=(0,0,0),
                lightAngle = (6, -30, -155), 
                strength = 2,
                shadowSoftness = 0.3):

    ## Initialize Blender scene
    blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

    ## read mesh
    mesh = readOBJ(filename, location, rotation, scale)

    ## set vertex colors
    mesh = setMeshScalars(mesh, vertex_scalars, color_map, color_type, cmin=cmin, cmax=cmax)

    ## set shading (uncomment one of them)
    # bpy.ops.object.shade_smooth() 

    # set material 
    meshVColor = colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    setMat_VColor(mesh, meshVColor)

    ## set invisible plane (shadow catcher)
    invisibleGround(shadowBrightness=0.9)

    ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
    cam = setCamera(camLocation, lookAtLocation, focalLength)

    ## set light
    sun = setLight_sun(lightPosition, lightAngle, strength, shadowSoftness)

    ## set ambient light
    # bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

    ## set gray shadow to completely white with a threshold 
    shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

    ## save blender file so that you can adjust parameters in the UI
    # blend_file_path = 'renderings/blend_files/' + output_name + '.blend'
    # if os.path.exists(blend_file_path):        
    #     os.remove(blend_file_path) # delete existing file
    # bpy.ops.wm.save_mainfile(filepath=blend_file_path)

    # save rendering
    os.makedirs(output_dir, exist_ok=True)
    outputPath = os.path.abspath(os.path.join(output_dir, output_name + '.png'))
    renderImage(outputPath, cam)


if __name__ == "__main__":
    if "--" in sys.argv:
        arg_start = sys.argv.index("--") + 1
        args = sys.argv[arg_start:]
    else:
        args = sys.argv[1:]

    if len(args) < 1:
        raise ValueError("Expected path to a .npy vertex scalar file")

    vertex_scalars_path = args[0]
    output_dir = args[1] if len(args) > 1 else './renderings/images'
    
    # Split the path and find the mesh name
    path_split = vertex_scalars_path.split('/')
    mesh_name = next((path_split[i+1] for i in range(len(path_split)) if '_results' in path_split[i]), None)

    if mesh_name == 'bunny':
        filename = './data/bunny.obj'
        location = (0.53,0.13,0.91)
        rotation = (90,0,100) 
        scale = (1.1,1.1,1.1)
    elif mesh_name == 'spot':
        filename = './data/spot.obj'
        location = (0.69,0,0.72)
        rotation = (90,0,-144) 
        scale = (1,1,1)
    elif mesh_name == 'cat':
        filename = './data/cat.obj'
        location = (0.8,0,0.1)
        rotation = (90,0,110) 
        scale = (0.9,0.9,0.9)
    elif mesh_name == 'hand':
        filename = './data/hand.obj'
        location = (1.06,-0.17,0.2)
        rotation = (390,10,80) 
        scale = (2.7,2.7,2.7)
    elif mesh_name == 'heightfield':
        filename = './data/heightfield.obj'
        location = (0.25,0,0.79)
        rotation = (75,0,90) 
        scale = (0.75,0.75,0.75)
    elif mesh_name == 'ellipsoid':
        filename = './data/ellipsoid.obj'
        location = (-0.5,0.2,0.55)
        rotation = (0,0,140) 
        scale = (0.5,0.5,0.5)
    elif mesh_name == 'hammer':
        filename = './data/hammer.obj'
        location = (-1.11,-0.56,0.25)
        rotation = (90,0,120) 
        scale = (3,3,3)
    elif mesh_name == 'mushroom':
        filename = './data/mushroom.obj'
        location = (0.73,0.17,0.71)
        rotation = (45,0,120) 
        scale = (0.36,0.36,0.36)
    elif mesh_name == 'plane':
        filename = './data/plane.obj'
        location = (-0.3,0,0.3)
        rotation = (10,0,55) 
        scale = (0.2,0.2,0.2)
    elif mesh_name == 'duck':
        filename = './data/duck.obj'
        location = (0.15,0.1,0.79)
        rotation = (100,0,120) 
        scale = (1.3,1.3,1.3)
    elif mesh_name == 'dziuk':
        filename = './data/dziuk.obj'
        location = (0.6,0,0.85)
        rotation = (160,0,100) 
        scale = (0.8,0.8,0.8)
    elif mesh_name == 'springer':
        if 'fem' in vertex_scalars_path:
            num = vertex_scalars_path.split('_')[-1][:-4]
            filename = f'./data/springer_{num}.obj'
        else:
            filename = './data/springer.obj'
        location = (0.69,0,0.1)
        rotation = (90,0,220) 
        scale = (0.38,0.38,0.38)
    elif mesh_name == '2d':
        filename = './data/2d.obj'
        location = (0.91,0,0.96)
        rotation = (-115,0,90) 
        scale = (0.8,0.8,0.8)
    else:
        filename = f'./data/{mesh_name}.obj'
        location = (0,0,0.79)
        rotation = (90,0,120) 
        scale = (1,1,1)

    vertex_scalars = np.load(vertex_scalars_path)
    output_name = str(vertex_scalars_path.replace('/', '|')[:-4])

    # print(vertex_scalars[:, 0].shape)
    # vertex_scalars = vertex_scalars[0]
    # print(vertex_scalars.shape)

    if 'error' in vertex_scalars_path:
        arr = np.load(os.path.dirname(vertex_scalars_path) + '/min_max_err.npy')
        cmin = arr[0]
        cmax = arr[1]
    elif 'pred' in vertex_scalars_path or 'true' in vertex_scalars_path:
        arr = np.load(os.path.dirname(vertex_scalars_path) + '/min_max_pred.npy')
        cmin = arr[0]
        cmax = arr[1]
    else:
        cmin = np.min(vertex_scalars)
        cmax = np.max(vertex_scalars)

    vertex_scalars = vertex_scalars.flatten()
    orig_cmin = cmin
    orig_cmax = cmax

    if 'error' in vertex_scalars_path:
        color_map = 'red_error'
    else:
        if mesh_name == "cat":
            color_map = 'YlOrRd'
        else:
            color_map = 'YlGnBu'
            temp = cmin
            cmin = -cmax
            cmax = -temp
            vertex_scalars = -vertex_scalars

    render_mesh(output_name,
                vertex_scalars,
                filename,
                location,
                rotation,
                scale,
                cmin,
                cmax,
                output_dir=output_dir,
                color_map=color_map
                ) 
    
    if "error" in vertex_scalars_path:
        print(f"error cmin: {orig_cmin}, error cmax: {orig_cmax}")
    else:
        print(f"pred/true cmin: {orig_cmin}, pred/true cmax: {orig_cmax}")