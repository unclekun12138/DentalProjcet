
"""
CBCT 3D Volume Dataset for tooth segmentation
Supports NIfTI format (converted from DICOM)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


class CBCTDataset(Dataset):
    """
    CBCT volume dataset for tooth/jaw segmentation.
    Expects:
        root/
            imagesTr/  (image_001.nii.gz, ...)
            labelsTr/  (label_001.nii.gz, ...)
    """
    def __init__(self, root_dir, patch_size=(128, 128, 64), split='train', augment=True):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.split = split
        self.augment = augment and split == 'train'
        
        self.image_dir = os.path.join(root_dir, 'imagesTr')
        self.label_dir = os.path.join(root_dir, 'labelsTr')
        
        self.samples = []
        if os.path.exists(self.image_dir):
            for f in sorted(os.listdir(self.image_dir)):
                if f.endswith(('.nii.gz', '.nii')):
                    self.samples.append(f)
        
        print(f"[CBCT-{split}] Found {len(self.samples)} volumes")
    
    def __len__(self):
        return max(len(self.samples), 1)
    
    def __getitem__(self, idx):
        if len(self.samples) == 0:
            return self._get_synthetic(idx)
        
        img_path = os.path.join(self.image_dir, self.samples[idx])
        label_name = self.samples[idx].replace('image', 'label')
        label_path = os.path.join(self.label_dir, label_name)
        
        img = nib.load(img_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.int64) if os.path.exists(label_path) else np.zeros_like(img, dtype=np.int64)
        
        # Normalize
        img = self._normalize(img)
        
        # Random crop patch
        img, label = self._random_crop(img, label)
        
        if self.augment:
            img, label = self._augment(img, label)
        
        # Add channel dim: (D, H, W) -> (1, D, H, W)
        img = img[np.newaxis, ...]
        
        return {
            'image': torch.from_numpy(img),
            'label': torch.from_numpy(label),
        }
    
    def _normalize(self, img):
        # Clip to reasonable range and z-score
        img = np.clip(img, -1000, 3000)
        mean, std = img.mean(), img.std()
        if std > 0:
            img = (img - mean) / std
        return img.astype(np.float32)
    
    def _random_crop(self, img, label):
        d, h, w = img.shape
        pd, ph, pw = self.patch_size
        
        # If volume smaller than patch, pad
        if d < pd or h < ph or w < pw:
            pad_d = max(0, pd - d)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            img = np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            d, h, w = img.shape
        
        d_start = np.random.randint(0, d - pd + 1) if d > pd else 0
        h_start = np.random.randint(0, h - ph + 1) if h > ph else 0
        w_start = np.random.randint(0, w - pw + 1) if w > pw else 0
        
        img = img[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label = label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        return img, label
    
    def _augment(self, img, label):
        # Random flip
        if np.random.rand() > 0.5:
            img = img[:, :, ::-1].copy()
            label = label[:, :, ::-1].copy()
        # Random intensity shift
        img += np.random.uniform(-0.1, 0.1)
        return img, label
    
    def _get_synthetic(self, idx):
        """Synthetic CBCT volume for pipeline testing."""
        np.random.seed(idx)
        pd, ph, pw = self.patch_size
        img = np.random.randn(pd, ph, pw).astype(np.float32) * 0.5
        # Add a "tooth" region
        img[pd//2-5:pd//2+5, ph//2-5:ph//2+5, pw//2-5:pw//2+5] += 2.0
        label = np.zeros((pd, ph, pw), dtype=np.int64)
        label[pd//2-5:pd//2+5, ph//2-5:ph//2+5, pw//2-5:pw//2+5] = 1
        return {
            'image': torch.from_numpy(img[np.newaxis, ...]),
            'label': torch.from_numpy(label),
        }

