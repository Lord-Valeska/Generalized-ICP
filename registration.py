import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
try:
    import open3d as o3d
    visualize = True
except ImportError:
    print('To visualize you need to install Open3D. \n \t>> You can use "$ pip install open3d"')
    visualize = False
    
from utils import ICPVisualizer, load_point_cloud, view_point_cloud, quaternion_matrix, \
    quaternion_from_axis_angle, load_pcs_and_camera_poses, save_point_cloud, load_point_cloud_customized, \
    add_noise, downsample_pc
from pfh import PFH, SPFH, FPFH, get_pfh_correspondence, get_transform, get_error, get_correspondence, \
    get_chamfer_error
from segmentation import segment
from scene import load_models

def alignment(pc_target, pc_source):
    target = pc_target[:, :3]
    source = pc_source[:, :3]
    
    threshold = 1e-5
    k = 16
    r = 0.03
    pfh_source = FPFH(source, r, k, 2, 3, 0)
    pfh_target = FPFH(target, r, k, 2, 3, 0)
    
    align = False
    errors = []
    
    R_full = np.eye(3) 
    t_full = np.zeros(3) 

    # Initial transform
    print("Initial transform...")
    current = time.time()
    C = get_pfh_correspondence(pfh_target, pfh_source)
    R, t = get_transform(C)
    aligned = pfh_source.transform(R, t)
    end = time.time()
    print(f"Iteration time: {end - current}")
    R_full = R @ R_full 
    t_full = R @ t_full + t.flatten()
    error = get_error(C, R, t)
    errors.append(error)
    print(error)
    print("Starting ICP...")
    
    # Standard ICP
    for i in range(5):
        current = time.time()
        C = get_correspondence(pfh_target, pfh_source)
        R, t = get_transform(C)
        aligned = pfh_source.transform(R, t)
        end = time.time()
        print(f"Iteration time: {end - current}")
        R_full = R @ R_full 
        t_full = R @ t_full + t.flatten()
        error = get_error(C, R, t)
        errors.append(error)
        print(error)
        if len(errors) > 1:
            relative_change = abs(errors[-1] - errors[-2]) / errors[-2]
            if relative_change < threshold or error <= threshold:
                print(f"Converged at iteration {i} with relative change {relative_change:.6f}")
                # if error <= threshold:
                align = True
                break
    align = True
    return align, R_full, t_full

def register(path_to_pointcloud_files, visualize):
    classification, pc_objects = segment(path_to_pointcloud_files, visualize)
    pc_models = load_models(path_to_pointcloud_files)
    pc_models = pc_models[:2]
    
    classes = np.unique(classification)
    pc_separated = {cls: pc_objects[classification == cls] for cls in classes}
    
    results = []
    for cls, pc in pc_separated.items():
        for i, pc_model in enumerate(pc_models):
            aligned, R, t = alignment(pc, pc_model)
            if aligned:
                print(f"Aligned with model {i}")
                results.append([cls, pc_model, R, t])
                break
    print("All Aligned!")
    for result in results:
        pc_model = result[1]
        R = result[2]
        t = result[3]
        pc_model[:, :3] = pc_model[:, :3] @ R.T + t
        pc_model[:, 3:] = [1.0, 0.0, 0.0]
        pc_objects = np.concatenate([pc_objects, pc_model], axis=0)
    if visualize:
        print('Displaying filtered point cloud. Close the window to continue.')
        view_point_cloud(pc_objects)
            
    return results
        
if __name__ == '__main__':
    path_to_files = 'pointclouds'
    register(path_to_files, visualize)

