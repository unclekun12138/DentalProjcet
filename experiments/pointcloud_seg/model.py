
"""
Simple PointNet for tooth segmentation (baseline)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSeg(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Segmentation head
        self.conv4 = nn.Conv1d(1024 + 64, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1)
        
        x1 = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x2 = F.relu(self.bn2(self.conv2(x1)))  # (B, 128, N)
        x3 = F.relu(self.bn3(self.conv3(x2)))  # (B, 1024, N)
        
        global_feat = torch.max(x3, dim=2, keepdim=True)[0]  # (B, 1024, 1)
        global_feat = global_feat.repeat(1, 1, x.shape[2])  # (B, 1024, N)
        
        x = torch.cat([x1, global_feat], dim=1)  # (B, 1088, N)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)  # (B, num_classes, N)
        
        x = x.transpose(2, 1)  # (B, N, num_classes)
        return x

