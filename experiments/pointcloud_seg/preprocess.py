
"""
Point cloud preprocessing utilities
- Normal estimation
- Curvature calculation
- Voxel downsampling
- Outlier removal
"""
import numpy as np
import open3d as o3d


def load_point_cloud(file_path):
    """Load point cloud from file."""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def estimate_normals(pcd, k=30):
    """Estimate normals using KNN search."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k)
    return pcd


def voxel_downsample(pcd, voxel_size=0.5):
    """Downsample point cloud using voxel grid."""
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """Remove outlier points using statistical outlier removal."""
    cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return cleaned


def compute_curvature(pcd):
    """
    Compute curvature at each point using normal variation.
    Returns curvature values array.
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Simple curvature: based on normal angle variation in local neighborhood
    # This is a simplified version; production should use proper PCA-based curvature
    curvature = np.zeros(len(points))
    
    # Build KDTree for neighbor search
    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    for i in range(len(points)):
        _, idx, _ = kd_tree.search_knn_vector_3d(points[i], 10)
        local_normals = normals[idx]
        # Curvature = variation of normal directions
        normal_var = np.std(local_normals, axis=0).mean()
        curvature[i] = normal_var
    
    return curvature


def preprocess_pipeline(file_path, voxel_size=0.5, remove_outliers_flag=True):
    """
    Full preprocessing pipeline:
    1. Load
    2. Remove outliers (optional)
    3. Voxel downsample
    4. Estimate normals
    5. Compute curvature
    
    Returns: points (N,3), normals (N,3), curvature (N,)
    """
    pcd = load_point_cloud(file_path)
    
    if remove_outliers_flag:
        pcd = remove_outliers(pcd)
    
    pcd = voxel_downsample(pcd, voxel_size)
    pcd = estimate_normals(pcd)
    
    curvature = compute_curvature(pcd)
    
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    return {
        'points': points,
        'normals': normals,
        'curvature': curvature,
    }


if __name__ == '__main__':
    # Test with synthetic data
    print("Preprocessing module loaded OK")
    print("Functions: load_point_cloud, estimate_normals, voxel_downsample,")
    print("           remove_outliers, compute_curvature, preprocess_pipeline")

