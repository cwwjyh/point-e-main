#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


from PIL import Image
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.point_cloud import PointCloud
from plyfile import PlyData,PlyElement


from matplotlib import pyplot as plt

import numpy as np


import cv2



# Model prep
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base1B' # use base300M or base1B for better results
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))


print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()

print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))


#Sampler prep
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 3.0],
)



#Execution of interfaces
# Load an image to condition on.
img = Image.open('/remote-home/caiweiwei/volsdf-transformer_1/data/resize_DTU_256/scan63/image/000000.png')

# Produce a sample from the model.
samples = None
for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
    samples = x



#Display of 3D point cloud
pc = sampler.output_to_point_clouds(samples)[0]
# pc.save('/remote-home/caiweiwei/point-e-main/point_e/examples/pointcloud.npz/pointcloud_24_ori.npz')
fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

fig.show()
print('fig', fig)

# Produce a mesh (with vertex colors)
mesh = marching_cubes_mesh(
    pc=pc,
    model=model,
    batch_size=4096,
    grid_size=32, # increase to 128 for resolution used in evals
    progress=True,
)

# Write the mesh to a PLY file to import into some other program.
with open('mesh_ply/base_1B/63_256mesh.ply', 'wb') as f: #'wb'以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件
    mesh.write_ply(f)

with open('mesh_ply/base_1B/63_256mesh.ply', 'rb') as f: #'rb'以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。
    plydata = PlyData.read(f)
#
#
# print(plydata)


