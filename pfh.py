#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

class PFH(object):
    def __init__(self, pointcloud, radius, num_neighbors=8, div=2, num_features=4, percentile=0):
        self.pc = pointcloud # (N, 3)
        self.r = radius
        self.nneighbors = num_neighbors
        self.tree = None
        self.normals, self.curvatures = self.get_normals()
        self.div = div
        self.nfeatures = num_features
        self.percentile = percentile
        self.idx_featured = None
        self.idx_regular = None
        self.get_categorized_idx()
    
    def get_featured_idx(self):
        return self.idx_featured
    
    def get_regular_idx(self):
        return self.idx_regular
    
    def get_point(self, idx):
        return self.pc[idx]
    
    def get_all_points(self):
        return self.pc
    
    def get_size(self):
        return self.pc.shape[0]

    def pca_normal(self, X):
        """
        X - (N, 3)
        """
        n = X.shape[0]
        mu = np.mean(X, axis=0)
        X = X - mu
        Q = (X.T @ X) / (n - 1)
        U, sigma, V_T = np.linalg.svd(Q)
        curvature = sigma[2] / (sigma[0] + sigma[1] + sigma[2])
        normal = V_T.T[:, -1]
        return normal, curvature

    def get_kNNs(self, idx):
        """
        Get k-neighborhood defined by a sphere centered at point with self.radius in self.pc

        PARAMS:
        idx - point index in self.pc

        RETURN:
        ndarray of k points
        """
        if self.tree is None:
            self.tree = KDTree(self.pc)

        point = self.pc[idx]
        idx_within_radius = self.tree.query_ball_point(point, r=self.r)
        # Exclude the point itself
        idx_within_radius = [i for i in idx_within_radius if i != idx]
    
        if idx_within_radius is not None:
            if len(idx_within_radius) > self.nneighbors:
                points_within_radius = self.pc[idx_within_radius]
                tree_within_radius = KDTree(points_within_radius)
                distances, k_indices = tree_within_radius.query(point, k=self.nneighbors)
                neighbor_indices = np.asarray(idx_within_radius)[k_indices]
            else:
                neighbor_indices = np.asarray(idx_within_radius)
            # print(f"{idx}: {self.nneighbors}-neighborhood within radius {self.r}: {neighbor_indices}")
            return np.asarray(neighbor_indices)
        else:
            print("No points within radius")
            return None
        
    def get_normals(self):
        """
        Get normals for all points in self.pc. Re-orient vectors outward (away from mean).

        RETURN:
        ndarray of normals
        """
        normals = []
        curvatures = []
        N = self.pc.shape[0]
        mean = np.mean(self.pc, axis=0)
        for i in range(N):
            neighbor_indices = self.get_kNNs(i)
            neighbors = self.pc[neighbor_indices]
            normal, curvature = self.pca_normal(neighbors)
            v = mean - self.pc[i]
            v /= np.linalg.norm(v)
            if np.dot(normal, v) > 0:
                normal *= -1
            normals.append(normal)
            curvatures.append(curvature)
        return np.asarray(normals), np.asarray(curvatures)
    
    def get_categorized_idx(self):
        all_indices = np.arange(len(self.pc))
        if self.percentile == 0:
            self.idx_featured = all_indices
            self.idx_regular = np.array([])
        else:
            self.idx_featured = np.where(self.curvatures > np.percentile(self.curvatures, self.percentile))[0]
            self.idx_regular = np.setdiff1d(all_indices, self.idx_featured)

    def get_features(self, idx):
        """
        idx - point index in self.pc

        RETURN:
        ndarray of size (self.nneighbors, self.nfeatures).
        """
        neighbor_indices = self.get_kNNs(idx)
        combined_list = [idx] + list(neighbor_indices)
        points_idx = np.array(combined_list, dtype=int)
        points_idx_copy = points_idx.copy()
        features = []
        for i_idx in points_idx:
            p_i = self.pc[i_idx]
            n_i = self.normals[i_idx]
            points_idx_copy = points_idx_copy[1:]
            for j_idx in points_idx_copy:
                p_j = self.pc[j_idx]
                n_j = self.normals[j_idx]
                if np.arccos(np.dot(n_i, p_j - p_i)) <= np.arccos(np.dot(n_j, p_i - p_j)):
                    source_idx = i_idx
                    source_point = p_i
                    target_idx = j_idx
                    target_point = p_j
                else:
                    source_idx = j_idx
                    source_point = p_j
                    target_idx = i_idx
                    target_point = p_i
                d = np.linalg.norm(target_point - source_point)
                # Construct the Darboux frame
                u = self.normals[source_idx]
                v = np.cross((target_point - source_point), u)
                w = np.cross(u, v)
                # Construct the four features (including the distance)
                nt = self.normals[target_idx]
                f1 = np.dot(v, nt)
                f2 = d
                f3 = np.dot(u, (target_point - source_point) / d)
                f4 = np.arctan(np.dot(w, nt) / np.dot(u, nt))
                if self.nfeatures == 4:
                    features.append(np.array([f1, f2, f3, f4]))
                elif self.nfeatures == 3:
                    features.append(np.array([f1, f3, f4]))
        features = np.asarray(features)
        return features

    def get_feature_histogram(self, idx):
        """
        Implemented 2 div, inclduing d in the feature.
        Try 3 div without distance.
        """
        features = self.get_features(idx)
        r = 0
        if self.nfeatures == 4:
            r = np.median(features[:, 1])
        histogram = np.zeros(self.div ** self.nfeatures)
        s = self.get_threshold(r)
        for j in range(0, features.shape[0]):
            index = 0
            for i in range(0, self.nfeatures):
                index += self.step(s[i], features[j, i]) * (self.div ** (i))
            histogram[index] += 1
        histogram /= features.shape[0]
        return histogram

    def get_threshold(self, r=0):
        delta = 2. / self.div
        s1 = np.array([-1 + i * delta for i in range(1, self.div)])
        s3 = np.array([-1 + i * delta for i in range(1, self.div)])

        delta = np.pi / self.div
        s4 = np.array([-np.pi / 2 + i * delta for i in range(1, self.div)])

        if self.nfeatures == 3:
            s = np.array([s1, s3, s4])
        elif self.nfeatures == 4:
            s2 = np.array([r])
            s = np.array([s1, s2, s3, s4])
        return s
    
    def step(self, si, fi):
        if self.div == 2:
            result = 0 if fi < si[0] else 1
        elif self.div == 3:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            else:
                result = 2
        elif self.div == 4:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            elif fi >= si[1] and fi < si[2]:
                result = 2
            else:
                result = 3
        elif self.div == 5:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            elif fi >= si[1] and fi < si[2]:
                result = 2
            elif fi >= si[2] and fi < si[3]:
                result = 3
            else:
                result = 4
        return result
    
    def get_all_histograms(self):
        N = self.idx_featured.shape[0]
        histograms = np.zeros((N, self.div ** self.nfeatures))
        for i, idx in enumerate(self.idx_featured):
            histograms[i] = self.get_feature_histogram(idx)
        return histograms
    
    def transform(self, R, t):
        self.pc = (R @ self.pc.T + t).T
        self.tree = None
        self.normals, self.curvatures = self.get_normals()
        # self.get_categorized_idx()
        return self.pc

class SPFH(PFH):
    def get_features(self, idx):
        """
        Make sure self.nfeatures == 3.

        idx - point index in self.pc

        RETURN:
        ndarray of size (self.nneighbors, self.nfeatures).
        """
        neighbor_indices = self.get_kNNs(idx)
        features = []
        i_idx = idx
        p_i = self.pc[i_idx]
        n_i = self.normals[i_idx]
        for j_idx in neighbor_indices:
            p_j = self.pc[j_idx]
            n_j = self.normals[j_idx]
            if np.arccos(np.dot(n_i, p_j - p_i)) <= np.arccos(np.dot(n_j, p_i - p_j)):
                source_idx = i_idx
                source_point = p_i
                target_idx = j_idx
                target_point = p_j
            else:
                source_idx = j_idx
                source_point = p_j
                target_idx = i_idx
                target_point = p_i
            d = np.linalg.norm(target_point - source_point)
            # Construct the Darboux frame
            u = self.normals[source_idx]
            v = np.cross((target_point - source_point), u)
            w = np.cross(u, v)
            # Construct the four features (including the distance)
            nt = self.normals[target_idx]
            f1 = np.dot(v, nt)
            f2 = d
            f3 = np.dot(u, (target_point - source_point) / d)
            f4 = np.arctan(np.dot(w, nt) / np.dot(u, nt))
            if self.nfeatures == 4:
                features.append(np.array([f1, f2, f3, f4]))
            elif self.nfeatures == 3:
                features.append(np.array([f1, f3, f4]))
        features = np.asarray(features)
        return features

class FPFH(SPFH):
    def __init__(self, pointcloud, radius, num_neighbors=8, div=2, num_features=4, percentile=0):
        super().__init__(pointcloud, radius, num_neighbors, div, num_features)
        self.histogram = []
        for i in range(self.pc.shape[0]):
            self.histogram.append(self.get_feature_histogram(i))
        self.histogram = np.asarray(self.histogram)
    
    def get_all_histograms(self):
        N = self.idx_featured.shape[0]
        histograms = np.zeros((N, self.div ** self.nfeatures))
        for i, idx in enumerate(self.idx_featured):
            neighbor_indices = self.get_kNNs(idx)
            sum_SPF = np.zeros_like(self.histogram[0])
            for neighbor_idx in neighbor_indices:
                distance = np.linalg.norm(self.pc[neighbor_idx] - self.pc[idx])
                sum_SPF += (1 / distance) * self.histogram[neighbor_idx]
            histograms[i] = self.histogram[idx] + (1 / len(neighbor_indices)) * sum_SPF
        return histograms

# # Old version for correspondence
# def get_correspondence(pfh_source, pfh_target):
#     C = []
#     histogram_source = pfh_source.get_all_histograms()
#     histogram_target = pfh_target.get_all_histograms()
#     featured_idx_source = pfh_source.get_featured_idx()
#     regular_idx_source = pfh_source.get_regular_idx()
#     featured_idx_target = pfh_target.get_featured_idx()
#     regular_idx_target = pfh_target.get_regular_idx()
#     for i, idx in enumerate(featured_idx_source):
#         p = pfh_source.get_point(idx)
#         histogram_p = histogram_source[i]
#         epsilon = 1e-10
#         numerator = (histogram_target - histogram_p) ** 2
#         denominator = histogram_target + histogram_p + epsilon
#         chi_squared_distances = np.sum(numerator / denominator, axis=1)
#         min_dist = np.min(chi_squared_distances)
#         min_index = np.argmin(chi_squared_distances)
#         min_index = featured_idx_target[min_index]
#         q = pfh_target.get_point(min_index)
#         C.append([p, q])
#     C = np.asarray(C) # (N, 2, 3)
#     return C

# def get_transform(C):
#     # Step 0
#     Cp = C[:, 0, :] # (N, 3)
#     Cq = C[:, 1, :] # (N, 3)
#     # Step 1
#     p_bar = np.mean(Cp, axis=0)
#     q_bar = np.mean(Cq, axis=0)
#     X = Cp - p_bar
#     Y = Cq - q_bar
#     # Step 2
#     S = X.T @ Y
#     U, sigma, V_T = np.linalg.svd(S)
#     # Step 3
#     M = np.eye(3)
#     M[2, 2] = np.linalg.det(V_T.T @ U.T)
#     R = V_T.T @ M @ U.T
#     t = q_bar.reshape(3, 1) - R @ p_bar.reshape(3, 1)
#     return R, t

# def get_error(C, R, t):
#     Cp = C[:, 0, :] # (N, 3)
#     Cq = C[:, 1, :] # (N, 3)
#     errors = (((R @ Cp.T).T + t.reshape(1, 3)) - Cq) ** 2
#     return np.sum(errors)

# New version for missing points
def get_pfh_correspondence(pfh_source, pfh_target):
    C = {}
    histogram_source = pfh_source.get_all_histograms()
    histogram_target = pfh_target.get_all_histograms()
    featured_idx_source = pfh_source.get_featured_idx()
    regular_idx_source = pfh_source.get_regular_idx()
    featured_idx_target = pfh_target.get_featured_idx()
    regular_idx_target = pfh_target.get_regular_idx()
    for i, idx in enumerate(featured_idx_source):
        p = tuple(pfh_source.get_point(idx))
        histogram_p = histogram_source[i]
        epsilon = 1e-10
        numerator = (histogram_target - histogram_p) ** 2
        denominator = histogram_target + histogram_p + epsilon
        chi_squared_distances = np.sum(numerator / denominator, axis=1)
        min_dist = np.min(chi_squared_distances)
        min_index = np.argmin(chi_squared_distances)
        min_index = featured_idx_target[min_index]
        q = tuple(pfh_target.get_point(min_index))
        if q in C:
            if min_dist < C[q][1]:
                C[q] = [p, min_dist]
        else:
            C[q] = [p, min_dist]
    if len(regular_idx_source) != 0:
        assigned_target = list(C.keys())
        for i, idx in enumerate(regular_idx_source):
            p = tuple(pfh_source.get_point(idx))
            Q = pfh_target.get_all_points()
            tree = KDTree(Q)
            distances, indices = tree.query(p, k=len(Q))
            for i, idx in enumerate(indices):
                if tuple(Q[idx]) not in assigned_target:
                    min_index = idx
                    min_dist = distances[i]
                    break
            q = tuple(Q[min_index])
            if q in C:
                if min_dist < C[q][1]:
                    C[q] = [p, min_dist]
            else:
                C[q] = [p, min_dist]
    return C

def get_correspondence(pfh_source, pfh_target):
    num_source = pfh_source.get_size()
    num_target = pfh_target.get_size()
    C = {}
    Q = pfh_target.get_all_points()
    for i in range(num_source):
        p = tuple(pfh_source.get_point(i))
        tree = KDTree(Q)
        distance, idx = tree.query(p)
        q = tuple(Q[idx])
        if q in C:
            if distance < C[q][1]:
                C[q] = [p, distance]
        else:
            C[q] = [p, distance]
    # C = []
    # P = pfh_source.get_all_points()
    # Q = pfh_target.get_all_points()
    # for i in range(num_source):
    #     p = P[i]
    #     distances = np.linalg.norm(Q - p, axis=0)
    #     q = Q[np.argmin(distances)]
    #     C.append([p, q])
    # C = np.asarray(C).squeeze()
    return C
    
def get_transform(C):
    # Step 0
    Cp, Cq = get_pq(C)
    # Step 1
    p_bar = np.mean(Cp, axis=0)
    q_bar = np.mean(Cq, axis=0)
    X = Cp - p_bar
    Y = Cq - q_bar
    # Step 2
    S = X.T @ Y
    U, sigma, V_T = np.linalg.svd(S)
    # Step 3
    M = np.eye(3)
    M[2, 2] = np.linalg.det(V_T.T @ U.T)
    R = V_T.T @ M @ U.T
    t = q_bar.reshape(3, 1) - R @ p_bar.reshape(3, 1)
    return R, t

def get_error(C, R, t):
    Cp, Cq = get_pq(C)
    errors = (((R @ Cp.T).T + t.reshape(1, 3)) - Cq) ** 2
    return np.sum(errors)

def get_chamfer_error(pfh1, pfh2):
    pc1 = pfh1.get_all_points()
    pc2 = pfh2.get_all_points()
    
    kd_tree_1 = KDTree(pc1)
    kd_tree_2 = KDTree(pc2)
    
    distances_1_to_2, _ = kd_tree_1.query(pc2)
    distances_2_to_1, _ = kd_tree_2.query(pc1)
    
    chamfer_dist = np.mean(distances_1_to_2**2) + np.mean(distances_2_to_1**2)
    return chamfer_dist

def get_pq(C):
    if isinstance(C, dict):
        Cq = []
        Cp = []
        for key in C:
            Cq.append(key)
            Cp.append(C[key][0])
        Cq = np.asarray(Cq)
        Cp = np.asarray(Cp)
    elif isinstance(C, np.ndarray):
        Cp = C[:, 0, :] # (495, 3)
        Cq = C[:, 1, :] # (495, 3)
    
    return Cq, Cp