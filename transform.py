import numpy as np
import time
import os
from scipy.spatial import KDTree
try:
    import open3d as o3d
    visualize = True
except ImportError:
    print('To visualize you need to install Open3D. \n \t>> You can use "$ pip install open3d"')
    visualize = False
    
from utils import add_noise
    
def transform_bunny(pc_bunny):
    pc_bunny_xyz = pc_bunny[:, :3]
    R = np.array([
        [1, 0, 0],
        [0,  0, -1],
        [0,  1, 0]
    ])
    pc_bunny_xyz = pc_bunny_xyz @ R.T
    pc_bunny_xyz[:, 2] -= 0.029

    pc_bunny[:, :3] = pc_bunny_xyz

    return pc_bunny

def transform_armadillo(pc_armadillo):
    pc_armadillo_xyz = pc_armadillo[:, :3]
    R = np.array([
        [1, 0, 0],
        [0,  0, -1],
        [0,  1, 0]
    ])
    pc_armadillo_xyz = pc_armadillo_xyz @ R.T
    pc_armadillo_xyz[:, 2] += 0.06

    pc_armadillo[:, :3] = pc_armadillo_xyz

    return pc_armadillo

def transform_chair(pc_chair):
    pc_chair[:, 2] += 0.09
    return pc_chair
    
def random_transform(pc):
    pc_xyz = pc[:, :3]
    # Random translation in X and Y, bounded by [-6, 6]
    T_random = np.array([
        np.random.uniform(-0.4, 0.4),  
        np.random.uniform(-0.4, 0.4),  
        0                          
    ])
    # Random rotation angle in radians
    theta = np.random.uniform(0, 2 * np.pi)
    # Rotation matrix for Z-axis
    R_random = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    pc_xyz = pc_xyz @ R_random.T + T_random
    pc_xyz = np.asarray(add_noise(pc_xyz, 0.001))
    pc[:, :3] = pc_xyz
    return pc