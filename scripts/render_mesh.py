# # if you want to call the toolbox the old way with `blender -b -P demo_XXX.py`, then uncomment these two lines
import sys, os
import shutil
sys.path.append("./blender_utils/")
sys.path.append("./src/")
from siren import MLP
from readOBJ import readOBJ
from blenderInit import blenderInit
from setMeshScalars import setMeshScalars
from invisibleGround import invisibleGround
from setCamera import setCamera
from setLight_sun import setLight_sun
from shadowThreshold import shadowThreshold
from renderImage import renderImage
from colorObj import colorObj
from setMat_plastic import setMat_plastic
import bpy
import os
import numpy as np

cb_black = (0/255.0, 0/255.0, 236/255.0, 1)
cb_orange = (230/255.0, 159/255.0, 0/255.0, 1)
cb_skyBlue = (86/255.0, 180/255.0, 233/255.0, 1)
cb_green = (0/255.0, 158/255.0, 115/255.0, 1)
cb_yellow = (240/255.0, 228/255.0, 66/255.0, 1)
cb_blue = (0/255.0, 114/255.0, 178/255.0, 1)
cb_vermillion = (213/255.0, 94/255.0, 0/255.0, 1)
cb_purple = (204/255.0, 121/255.0, 167/255.0, 1)

def render_mesh(output_name,
                filename,
                location,
                rotation,
                scale,
                mesh_color=cb_purple,
                imgRes_x = 1080, 
                imgRes_y = 1080, 
                numSamples = 200,
                exposure = 1.5,
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

    ## set shading (uncomment one of them)
    # bpy.ops.object.shade_smooth() 

    # set material 
    meshColor = colorObj(mesh_color, 0.5, 1.0, 1.0, 0.0, 2.0)
    setMat_plastic(mesh, meshColor)

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

    # ## save blender file so that you can adjust parameters in the UI
    # blend_file_path = 'renderings/blend_files/' + output_name + '.blend'
    # if os.path.exists(blend_file_path):        
    #     os.remove(blend_file_path) # delete existing file
    # bpy.ops.wm.save_mainfile(filepath=blend_file_path)

    # save rendering
    output_dir_renderings = os.path.abspath('./renderings/images')
    output_dir_scripts = os.path.abspath('./scripts/images')
    os.makedirs(output_dir_renderings, exist_ok=True)
    os.makedirs(output_dir_scripts, exist_ok=True)

    output_path_renderings = os.path.join(output_dir_renderings, output_name + '.png')
    output_path_scripts = os.path.join(output_dir_scripts, output_name + '.png')

    renderImage(output_path_renderings, cam)
    shutil.copy2(output_path_renderings, output_path_scripts)


def _mesh_params(mesh_name):
    if mesh_name == 'bunny':
        return './data/bunny.obj', (0.53,0.13,0.91), (90,0,100), (1.1,1.1,1.1)
    if mesh_name == 'spot':
        return './data/spot.obj', (0.69,0,0.72), (90,0,-144), (1,1,1)
    if mesh_name == 'cylinder':
        return './data/cylinder.obj', (0.5,0,0.79), (90,0,120), (0.5,0.5,0.5)
    if mesh_name == 'cylinder_minimal_surface':
        return './data/cylinder_minimal_surface.obj', (0.5,0,0.79), (90,0,120), (0.5,0.5,0.5)
    if mesh_name == 'hand':
        return './data/hand.obj', (1.06,-0.17,0.2), (390,10,80), (2.7,2.7,2.7)
    if mesh_name == 'heightfield':
        return './data/heightfield.obj', (0.25,0,0.79), (75,0,90), (0.75,0.75,0.75)
    if mesh_name == 'ellipsoid':
        return './data/ellipsoid.obj', (-0.5,0.2,0.55), (0,0,140), (0.5,0.5,0.5)
    if "moai" in mesh_name:
        return f'./data/{mesh_name}.obj', (1.15,0.06,0.75), (90,-180,90), (2,2,2)
    return f'./data/{mesh_name}.obj', (0,0,0.79), (90,0,120), (1,1,1)


if __name__ == "__main__":
    mesh_name = sys.argv[1]

    if mesh_name in {'cylinder', 'cylinder_minimal_surface'}:
        render_jobs = [
            ('cylinder', cb_green),
            ('cylinder_minimal_surface', cb_orange),
        ]
    else:
        render_jobs = [(mesh_name, cb_purple)]

    for job_mesh_name, job_color in render_jobs:
        filename, location, rotation, scale = _mesh_params(job_mesh_name)
        output_name = str(job_mesh_name + '_solid')
        render_mesh(
            output_name,
            filename,
            location,
            rotation,
            scale,
            mesh_color=job_color,
        )
    