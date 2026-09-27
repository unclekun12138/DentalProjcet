
"""
Training script for point cloud tooth segmentation
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
from dataset import TeethPointCloudDataset
from model import PointNetSeg


def compute_iou(pred, target, num_classes=11):
    """Compute mean IoU."""
    ious = []
    for c in range(num_classes):
        intersection = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


def train():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_classes = 11  # 0=gum, 1-10=teeth
    batch_size = 8
    epochs = 50
    lr = 0.001
    
    # Data
    train_dataset = TeethPointCloudDataset(
        root_dir='/root/autodl-tmp/project/data/raw/teeth3ds',
        n_points=20000,
        split='train',
        augment=True
    )
    val_dataset = TeethPointCloudDataset(
        root_dir='/root/autodl-tmp/project/data/raw/teeth3ds',
        n_points=20000,
        split='val',
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = PointNetSeg(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    best_miou = 0.0
    save_dir = '/root/autodl-tmp/project/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(points)  # (B, N, num_classes)
            loss = criterion(outputs.reshape(-1, num_classes), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_iou = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(points)
                pred = outputs.argmax(dim=-1)
                val_iou += compute_iou(pred.cpu(), labels.cpu(), num_classes)
                val_batches += 1
        
        avg_loss = train_loss / len(train_loader)
        avg_iou = val_iou / val_batches
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val mIoU: {avg_iou:.4f}")
        
        if avg_iou > best_miou:
            best_miou = avg_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': best_miou,
            }, os.path.join(save_dir, 'pointnet_best.pth'))
    
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    print(f"Model saved to {save_dir}/pointnet_best.pth")


if __name__ == '__main__':
    train()

