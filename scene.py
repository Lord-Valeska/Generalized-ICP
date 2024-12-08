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

import transform
from utils import ICPVisualizer, load_point_cloud, view_point_cloud, quaternion_matrix, \
    quaternion_from_axis_angle, load_pcs_and_camera_poses, save_point_cloud, load_point_cloud_customized, \
    add_noise, downsample_pc
    


def transform_point_cloud(point_cloud, t, R):
    """
    Transform a point cloud applying a rotation and a translation
    :param point_cloud: np.arrays of size (N, 6)
    :param t: np.array of size (3,) representing a translation.
    :param R: np.array of size (3,3) representing a 3D rotation matrix.
    :return: np.array of size (N,6) resulting in applying the transformation (t,R) on the point cloud point_cloud.
    """
    # ------------------------------------------------
    position = point_cloud[:, :3]
    transformed_position = R @ position.T + t.reshape(3, 1)
    transformed_point_cloud = np.hstack([transformed_position.T, point_cloud[:, 3:]])
    # ------------------------------------------------
    return transformed_point_cloud

def merge_point_clouds(point_clouds, camera_poses):
    """
    Register multiple point clouds into a common reference and merge them into a unique point cloud.
    :param point_clouds: List of np.arrays of size (N_i, 6)
    :param camera_poses: List of tuples (t_i, R_i) representing the camera i pose.
              - t: np.array of size (3,) representing a translation.
              - R: np.array of size (3,3) representing a 3D rotation matrix.
    :return: np.array of size (N, 6) where $$N = sum_{i=1}^K N_i$$
    """
    # ------------------------------------------------
    transformed_pcs = []
    for pc, (t, R) in zip(point_clouds, camera_poses):
        transformed_pc = transform_point_cloud(pc, t, R)
        transformed_pcs.append(transformed_pc)
    merged_point_cloud = np.vstack(transformed_pcs)
    # ------------------------------------------------
    return merged_point_cloud

def filter_point_cloud(point_cloud):
    """
    Remove unnecessary point given the scene point_cloud.
    :param point_cloud: np.array of size (N,6)
    :return: np.array of size (n,6) where n <= N
    """
    # ------------------------------------------------
    min_x, max_x = (-0.45, 0.45)
    min_y, max_y = (-0.45, 0.45)
    filtered_pc = point_cloud[
        (point_cloud[:, 0] >= min_x) & (point_cloud[:, 0] <= max_x) &
        (point_cloud[:, 1] >= min_y) & (point_cloud[:, 1] <= max_y)
    ]
    # ------------------------------------------------
    return filtered_pc

def load_models(path_to_pointcloud_files):
    color = [0, 100 / 255, 0]
    # Load the model
    pc_M = load_point_cloud_customized("pointclouds/michigan_M_med.ply", "M", color, 0.0034, 1) # Model
    # pc_M = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Model
    pc_M[:, 3:] = np.array([.73, .21, .1]) * np.ones((pc_M.shape[0], 3)) # Paint it red
    pc_bunny = load_point_cloud_customized("pointclouds/bun_zipper.ply", "Bunny", color, 0.005, 1) # Model
    pc_armadillo = load_point_cloud_customized("pointclouds/Armadillo.ply", "Armadillo", color, 3.7, 0.001) # Model
    pc_chair = load_point_cloud_customized("pointclouds/chair.ply", "Chair", color, 20, 0.0002) # Model
    return [pc_M, pc_bunny, pc_armadillo, pc_chair]
    
def scene_construction(path_to_pointcloud_files, visualize=True):
    color = [0, 100 / 255, 0]
    
    pc_models = load_models(path_to_pointcloud_files)

    # Generate the scene
    pcs, camera_poses = load_pcs_and_camera_poses(path_to_pointcloud_files)
    pc = merge_point_clouds(pcs, camera_poses)
    pc_M_scene = filter_point_cloud(pc)
    # pc_M_scene[:, -3:] = color

    pc_bunny_scene = downsample_pc(transform.random_transform(transform.transform_bunny(pc_models[1])), 2000)
    pc_armadillo_scene = downsample_pc(transform.random_transform(transform.transform_armadillo(pc_models[2])), 2000)
    pc_chair_scene = downsample_pc(transform.random_transform(transform.transform_chair(pc_models[3])), 2000)

    pc_scene = np.concatenate([pc_M_scene, pc_bunny_scene], axis=0)
    np.random.shuffle(pc_scene)

    if visualize:
        print('Displaying scene point cloud. Close the window to continue.')
        view_point_cloud(pc_scene)
    
    return pc_scene

# if __name__ == '__main__':
#     path_to_files = 'pointclouds'
#     scene_construction(path_to_files, visualize=visualize)
