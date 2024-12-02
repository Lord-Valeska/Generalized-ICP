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

from utils import ICPVisualizer, load_point_cloud, view_point_cloud, quaternion_matrix, \
    quaternion_from_axis_angle, load_pcs_and_camera_poses, save_point_cloud, load_point_cloud_customized, \
    add_noise

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
    min_x, max_x = (-0.6, 0.6)
    min_y, max_y = (-0.6, 0.6)
    filtered_pc = point_cloud[
        (point_cloud[:, 0] >= min_x) & (point_cloud[:, 0] <= max_x) &
        (point_cloud[:, 1] >= min_y) & (point_cloud[:, 1] <= max_y)
    ]
    # ------------------------------------------------
    return filtered_pc

def random_transform_bunny(pc_bunny):
    pc_bunny_xyz = pc_bunny[:, :3]
    R = np.array([
        [1, 0, 0],
        [0,  0, -1],
        [0,  1, 0]
    ])
    pc_bunny_xyz = pc_bunny_xyz @ R.T
    pc_bunny_xyz[:, 2] -= 0.029

    # Random translation in X and Y, bounded by [-6, 6]
    T_random = np.array([
        np.random.uniform(-0.5, 0.5),  
        np.random.uniform(-0.5, 0.5),  
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
    pc_bunny_xyz = pc_bunny_xyz @ R_random.T + T_random
    pc_bunny_xyz = np.asarray(add_noise(pc_bunny_xyz, 0.002))
    print(pc_bunny_xyz)

    pc_bunny[:, :3] = pc_bunny_xyz

    return pc_bunny

def scene_construction(path_to_pointcloud_files, visualize=True):
    color = [0, 100 / 255, 0]
    # Load the model
    pc_M = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Model
    pc_M[:, 3:] = np.array([.73, .21, .1]) * np.ones((pc_M.shape[0], 3)) # Paint it red
    pc_bunny = load_point_cloud_customized("pointclouds/bun_zipper.ply", color, 0.005) # Model

    # Generate the scene
    pcs, camera_poses = load_pcs_and_camera_poses(path_to_pointcloud_files)
    pc = merge_point_clouds(pcs, camera_poses)
    pc_M_scene = filter_point_cloud(pc)
    # pc_M_scene[:, -3:] = color

    pc_bunny_scene = random_transform_bunny(pc_bunny)

    pc_scene = np.concatenate([pc_M_scene, pc_bunny_scene], axis=0)

    if visualize:
        print('Displaying filtered point cloud. Close the window to continue.')
        view_point_cloud(pc_scene)
    else:
        print('Filtered scene point clouds saved as we cannot visualize it.\n Use software such as Meshlab to visualize them.')
        save_point_cloud(pc_scene, 'filtered_scene_pc', path_to_pointcloud_files)

if __name__ == '__main__':
    path_to_files = 'pointclouds'
    scene_construction(path_to_files, visualize=visualize)
