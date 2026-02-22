# Copyright 2020 Hsueh-Ti Derek Liu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bpy

def setMat_VColor(mesh, meshVColor):
    # Ensure the mesh has vertex color layers
    if not mesh.data.vertex_colors:
        mesh.data.vertex_colors.new(name="Col")  # Creates a new vertex color layer if it doesn't exist

    # Create a new material
    mat = bpy.data.materials.new(name='MeshMaterial')
    mesh.data.materials.append(mat)  # Assign the material to the mesh
    mesh.active_material = mat
    mat.use_nodes = True  # Enable nodes for the material
    tree = mat.node_tree  # Get the material's node tree
    
    # Add a node for the vertex color
    attribute_node = tree.nodes.new('ShaderNodeAttribute')
    attribute_node.attribute_name = "Col"  # Use the "Col" attribute (vertex color layer)
    
    # Add a Hue/Saturation node to adjust the colors
    HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
    tree.links.new(attribute_node.outputs['Color'], HSVNode.inputs['Color'])
    HSVNode.inputs['Saturation'].default_value = meshVColor.S  # Set saturation
    HSVNode.inputs['Value'].default_value = meshVColor.V  # Set value
    HSVNode.inputs['Hue'].default_value = meshVColor.H  # Set hue
    HSVNode.location.x -= 200  # Move the node left for visibility

    # Set Brightness and Contrast adjustment
    BCNode = tree.nodes.new('ShaderNodeBrightContrast')
    BCNode.inputs['Bright'].default_value = meshVColor.B  # Set brightness
    BCNode.inputs['Contrast'].default_value = meshVColor.C  # Set contrast
    BCNode.location.x -= 400  # Move the node left for visibility
    
    # Link the nodes: Attribute -> HSV -> Brightness/Contrast -> Principled BSDF
    tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
    
    # Set the Principled BSDF shader's settings (just roughness here)
    principled_bsdf_node = tree.nodes.get("Principled BSDF")
    if principled_bsdf_node:
        principled_bsdf_node.inputs['Roughness'].default_value = 1.0
        principled_bsdf_node.inputs['Sheen Tint'].default_value = [0, 0, 0, 1]
    
    # Link Brightness/Contrast node to the Base Color of the Principled BSDF
    tree.links.new(BCNode.outputs['Color'], principled_bsdf_node.inputs['Base Color'])

