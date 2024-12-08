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
    
from scene import scene_construction
from sklearn.cluster import DBSCAN
    
def load_scene(path_to_files, visualize):
    pc_scene = scene_construction(path_to_files, visualize)
    return pc_scene

def ransac(data):
    #Fit a plane to the data using ransac
    K = 100
    e_threshold = 0.02
    N = 150
    e_best = np.inf
    model_best = None

    def fit(X):
        n = data.shape[0]
        mu = np.mean(data, axis=0)
        X = X - mu
        Q = (X.T @ X) / (n - 1)
        U, sigma, V_T = np.linalg.svd(Q)
        normal = V_T[-1].T
        d = - normal.T @ mu
        return (np.matrix(normal), d, np.matrix(mu))
    
    def error(data, model):
        residuals = np.array(data @ model[0].T + model[1])
        errors = residuals ** 2
        return errors.squeeze()

    for i in range(10):
        random_indices = np.random.choice(data.shape[0], size=3, replace=False)
        remaining_indices = np.setdiff1d(np.arange(data.shape[0]), random_indices)
        R = data[random_indices]
        remaining_data = data[remaining_indices] 
        model = fit(R)
        es = error(remaining_data, model)
        C = remaining_data[es < e_threshold]
        if C.shape[0] > N:
            RUC = np.concatenate((R, C), axis=0)
            model = fit(RUC)
            es = error(RUC, model)
            e_total = np.sum(es)
            if e_total < e_best:
                e_best = e_total
                model_best = model

    normal, d, mu = model_best # normal: 1x3
    print(f"Plane equation: {normal[0, 0]}x + {normal[0, 1]}y + {normal[0, 2]}z + ({d}) = 0")
    return model_best
    
def filter_table(pc, model):
    distance_threshold = 0.01
    normal, d, mu = model
    numerator = np.abs(np.dot(pc, normal.T) + d)
    denominator = np.linalg.norm(normal[0])
    distances = numerator / denominator
    filtered_idx = np.where(distances > distance_threshold)[0]
    return filtered_idx
    
def segment(path_to_files, visualize):
    eps = 0.02
    min_points = 10
    
    pc_scene = load_scene(path_to_files, visualize=visualize)
    pc_scene_xyz = pc_scene[:, :3]
    model = ransac(pc_scene_xyz)
    filtered_idx = filter_table(pc_scene_xyz, model)
    pc_objects = pc_scene[filtered_idx]
    if visualize:
        print('Displaying filtered point cloud. Close the window to continue.')
        view_point_cloud(pc_objects)
     
    dbscan = DBSCAN(eps=eps, min_samples=min_points)
    dbscan.fit(pc_objects[:, :3])
    classifications = dbscan.labels_
    classes = list(set(classifications))
    print(classes)
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))  # Use a colormap
    color_mapping = {value: color for value, color in zip(classes, colors)}
    color_mapping[-1] = [0, 0, 0, 1]
    new_colors = np.array([color_mapping[x] for x in classifications], dtype=float)
    pc_objects[:, 3:] = new_colors[:, :3]
    if visualize:
        print('Displaying filtered point cloud. Close the window to continue.')
        view_point_cloud(pc_objects)
    return classifications, pc_objects
        
# if __name__ == '__main__':
#     path_to_files = 'pointclouds'
#     segment(path_to_files, visualize)
    
    
    