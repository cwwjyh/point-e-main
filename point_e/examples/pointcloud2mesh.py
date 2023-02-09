#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from plyfile import PlyData,PlyElement

from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()

print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))


# In[ ]:


# Load a point cloud we want to convert into a mesh.
pc = PointCloud.load('/remote-home/caiweiwei/point-e-main/point_e/examples/pointcloud.npz/pointcloud_24_ori.npz')

# Plot the point cloud as a sanity check.
fig = plot_point_cloud(pc, grid_size=2)
fig.show()


# In[ ]:


# Produce a mesh (with vertex colors)
mesh = marching_cubes_mesh(
    pc=pc,
    model=model,
    batch_size=4096,
    grid_size=32, # increase to 128 for resolution used in evals
    progress=True,
)


# In[ ]:


# Write the mesh to a PLY file to import into some other program.
with open('mesh_ply/24_orimesh.ply', 'wb') as f: #'wb'以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件
    mesh.write_ply(f)

with open('mesh_ply/24_orimesh.ply', 'rb') as f: #'rb'以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。
    plydata = PlyData.read(f)


print(plydata)

