import numpy as np
import time
import os
import copy
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
    add_noise, downsample_pc, get_o3d_pc
from segmentation import segment
from scene import load_models

def alignment(pcd_scene, pcd_model):
    num = 0.01
    print(f"Number of points in model: {len(pcd_model.points)}")
    print(f"Number of points in scene: {len(pcd_scene.points)}")
    
    # Estimate normals
    pcd_model.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=num * 2, max_nn=50))
    pcd_scene.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=num * 2, max_nn=50))
    
    # Compute FPFH features
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_model,
                o3d.geometry.KDTreeSearchParamHybrid(radius=num * 5, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_scene,
                o3d.geometry.KDTreeSearchParamHybrid(radius=num * 5, max_nn=100)
    )

    # Initial transform with RANSAC
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=pcd_model,  
        target=pcd_scene,  
        source_feature=source_fpfh,  
        target_feature=target_fpfh,  
        mutual_filter=False,  
        max_correspondence_distance=0.4, 
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,  # Number of points for RANSAC
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.008)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1.0)
    )
    
    # Generalized ICP
    # Using the transformation from RANSAC as the initial transformation
    result_icp = o3d.pipelines.registration.registration_generalized_icp(
        pcd_model, pcd_scene, max_correspondence_distance=0.3,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    )

    # # Point-to-Point ICP
    # result_icp = o3d.pipelines.registration.registration_icp(
    #     pcd_model, pcd_scene, max_correspondence_distance=0.3,
    #     init=result_ransac.transformation,
    #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # )
    # # Slight offset with M

    # # Point-to-Plane ICP
    # result_icp = o3d.pipelines.registration.registration_icp(
    #     pcd_model, pcd_scene, max_correspondence_distance=0.3,
    #     init=result_ransac.transformation,
    #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    # )
    
    return result_icp
    
def register(path_to_pointcloud_files, visualize):
    classification, pc_objects = segment(path_to_pointcloud_files, visualize)
    pcd_objects = get_o3d_pc(pc_objects)
    pcd_models = load_models(path_to_pointcloud_files)
    for i in range(len(pcd_models)):
        model = pcd_models[i]
        pcd_models[i] = get_o3d_pc(model)
    
    # Separate the object points based on classification
    classes = np.unique(classification)
    pcd_separated = {}
    for cls in classes:
        if cls != -1:
            class_points = pc_objects[classification == cls]
            pcd_separated[cls] = get_o3d_pc(class_points)  
    
    results = []
    for cls, pcd in pcd_separated.items():
        results_per_cls = []
        for i, pcd_model in enumerate(pcd_models):
            print(f"Starting alignment between target and model {i}")
            current = time.time()
            result_per_cls = alignment(pcd, pcd_model)
            end = time.time()
            print(f"Alignment done for model {i} in {end - current} seconds")
            results_per_cls.append({
            "model": pcd_model,
            "fitness": result_per_cls.fitness,
            "rmse": result_per_cls.inlier_rmse,
            "transformation": result_per_cls.transformation
            })
        best_result_per_cls = max(results_per_cls, key=lambda x: (x['fitness'], -x['rmse']))
        results.append(best_result_per_cls)
    print("All Aligned!")
    
    visualization_geometries = []
    if isinstance(pcd_objects, list):
        visualization_geometries.extend(pcd_objects)
    else:
        visualization_geometries.append(pcd_objects)
        
    for result in results:
        print(f"Best fitness: {result['fitness']}, best RMSE: {result['rmse']}")
        if result['rmse'] < 0.015: # 0.02 for gicp, 0.01 for point to point and point to plane
            model_best = copy.deepcopy(result["model"])
            model_best.transform(result["transformation"])
            visualization_geometries.append(model_best)
        
    print('Displaying filtered point cloud. Close the window to continue.')
    o3d.visualization.draw_geometries(visualization_geometries)
            
if __name__ == '__main__':
    path_to_files = 'pointclouds'
    register(path_to_files, visualize)

