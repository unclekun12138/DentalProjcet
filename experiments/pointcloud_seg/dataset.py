
"""
Point Cloud Dataset for Teeth Segmentation
Supports Teeth3DS+ format and synthetic data for pipeline testing
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d


class TeethPointCloudDataset(Dataset):
    """
    Teeth point cloud dataset for tooth segmentation.
    Expects directory structure:
        root/
            scans/  (ply/obj files)
            labels/ (corresponding label files)
    """
    def __init__(self, root_dir, n_points=20000, split='train', augment=True):
        self.root_dir = root_dir
        self.n_points = n_points
        self.split = split
        self.augment = augment and split == 'train'
        
        self.scan_dir = os.path.join(root_dir, 'scans')
        self.label_dir = os.path.join(root_dir, 'labels')
        
        self.samples = []
        if os.path.exists(self.scan_dir):
            for f in sorted(os.listdir(self.scan_dir)):
                if f.endswith(('.ply', '.obj', '.pcd')):
                    self.samples.append(f)
        
        print(f"[{split}] Found {len(self.samples)} scans")
    
    def __len__(self):
        return max(len(self.samples), 1)
    
    def __getitem__(self, idx):
        if len(self.samples) == 0:
            # Return synthetic data for pipeline testing
            return self._get_synthetic(idx)
        
        scan_path = os.path.join(self.scan_dir, self.samples[idx])
        pcd = o3d.io.read_point_cloud(scan_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Load labels if available
        label_name = self.samples[idx].replace('.ply', '.npy').replace('.obj', '.npy')
        label_path = os.path.join(self.label_dir, label_name)
        if os.path.exists(label_path):
            labels = np.load(label_path).astype(np.int64)
        else:
            labels = np.zeros(len(points), dtype=np.int64)
        
        # Normalize
        points = self._normalize(points)
        
        # Random sample n_points
        if len(points) >= self.n_points:
            choice = np.random.choice(len(points), self.n_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.n_points, replace=True)
        points = points[choice]
        labels = labels[choice]
        
        if self.augment:
            points = self._augment(points)
        
        return {
            'points': torch.from_numpy(points),
            'labels': torch.from_numpy(labels),
        }
    
    def _normalize(self, points):
        center = points.mean(axis=0)
        points -= center
        scale = np.max(np.abs(points))
        if scale > 0:
            points /= scale
        return points.astype(np.float32)
    
    def _augment(self, points):
        # Random rotation around z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
        points = points @ R.T
        # Random jitter
        points += np.random.normal(0, 0.01, points.shape).astype(np.float32)
        return points
    
    def _get_synthetic(self, idx):
        """Generate synthetic tooth-like point cloud for pipeline testing."""
        np.random.seed(idx)
        n = self.n_points
        # Simulate: gum (class 0) + several teeth (class 1-10)
        n_teeth = 8
        points_list = []
        labels_list = []
        
        # Gum: flat surface
        gum_pts = np.random.randn(n // 2, 3).astype(np.float32) * 0.3
        gum_pts[:, 2] = np.abs(gum_pts[:, 2]) * 0.1
        points_list.append(gum_pts)
        labels_list.append(np.zeros(len(gum_pts), dtype=np.int64))
        
        # Teeth: small bumps
        for i in range(n_teeth):
            cx = (i - n_teeth/2) * 0.25
            tooth_pts = np.random.randn(n // (n_teeth*2), 3).astype(np.float32) * 0.08
            tooth_pts[:, 0] += cx
            tooth_pts[:, 2] += 0.1
            points_list.append(tooth_pts)
            labels_list.append(np.full(len(tooth_pts), i+1, dtype=np.int64))
        
        points = np.concatenate(points_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        choice = np.random.choice(len(points), n, replace=True)
        points = points[choice]
        labels = labels[choice]
        
        return {
            'points': torch.from_numpy(points),
            'labels': torch.from_numpy(labels),
        }

