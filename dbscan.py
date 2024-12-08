import numpy as np
import math

UNCLASSIFIED = False
NOISE = None

def dist(p, q):
    return math.sqrt(np.power(p-q,2).sum())

def eps_neighborhood(p, q, eps):
    return dist(p,q) < eps

def region_query(data, point_id, eps):
    n_points = data.shape[0]
    seeds = []
    for i in range(0, n_points):
        if eps_neighborhood(data[point_id], data[i], eps):
            seeds.append(i)
    return seeds

def expand_cluster(data, classifications, point_id, cluster_id, eps, min_points):
    seeds = region_query(data, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = region_query(data, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True
    
def dbscan(data, eps, min_points):
    cluster_id = 1
    n_points = data.shape[0]
    classifications = [UNCLASSIFIED] * n_points
    print(n_points)
    for point_id in range(0, n_points):
        if point_id % 1000 == 0:
            print(point_id)
        point = data[point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if expand_cluster(data, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications