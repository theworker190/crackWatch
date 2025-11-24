import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import mobilenet_v3_small


# ============================================================================
# Dataset
# ============================================================================
class CrackSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, invert_mask=False):
        """
        Args:
            image_dir: folder containing crack images
            mask_dir: folder containing binary masks (same filenames)
            transform: albumentations transform
            invert_mask: if True, inverts mask (for datasets where cracks are black)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.invert_mask = invert_mask
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # Invert if needed (for datasets where cracks are black)
        if self.invert_mask:
            mask = 255 - mask
        
        # Binarize mask: 0 or 1
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask


# ============================================================================
# MobileNetV3-UNet Model
# ============================================================================
class MobileNetV3UNet(nn.Module):
    def __init__(self, out_channels=1, pretrained=True):
        super(MobileNetV3UNet, self).__init__()
        
        # Load pretrained MobileNetV3 as encoder
        from torchvision.models import MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = mobilenet_v3_small(weights=weights)
        self.features = mobilenet.features
        
        # Extract encoder layers at different depths for skip connections
        # MobileNetV3-Small has 12 inverted residual blocks
        self.enc1 = nn.Sequential(*self.features[0:2])   # 16 channels
        self.enc2 = nn.Sequential(*self.features[2:4])   # 24 channels
        self.enc3 = nn.Sequential(*self.features[4:9])   # 48 channels
        self.enc4 = nn.Sequential(*self.features[9:12])  # 96 channels
        
        # Bottleneck
        self.bottleneck = nn.Sequential(*self.features[12:])  # 576 channels
        
        # Decoder with upsampling
        self.upconv4 = nn.ConvTranspose2d(576, 96, 2, stride=2)
        self.dec4 = self.conv_block(96 + 96, 96)
        
        self.upconv3 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.dec3 = self.conv_block(48 + 48, 48)
        
        self.upconv2 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.dec2 = self.conv_block(24 + 24, 24)
        
        self.upconv1 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.dec1 = self.conv_block(16 + 16, 16)
        
        # Final upsampling to match input resolution
        self.final_up = nn.ConvTranspose2d(16, 16, 2, stride=2)
        
        # Output layer
        self.out = nn.Conv2d(16, out_channels, 1)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)          # 1/2 resolution
        enc2 = self.enc2(enc1)       # 1/4 resolution
        enc3 = self.enc3(enc2)       # 1/8 resolution
        enc4 = self.enc4(enc3)       # 1/16 resolution
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # 1/32 resolution
        
        # Decoder with skip connections (match spatial dimensions)
        dec4 = self.upconv4(bottleneck)
        # Match spatial dimensions if needed
        if dec4.shape[2:] != enc4.shape[2:]:
            dec4 = nn.functional.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = nn.functional.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = nn.functional.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = nn.functional.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final upsampling to original resolution
        final = self.final_up(dec1)
        out = self.out(final)
        
        # Ensure output matches input size
        if out.shape[2:] != x.shape[2:]:
            out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out


# ============================================================================
# Loss Functions
# ============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)  # Add channel dimension
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
    
    return running_loss / len(loader)


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union (IoU) metric"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()


# ============================================================================
# Main Training Script
# ============================================================================
def main():
    # Configuration
    TRAIN_IMG_DIR = "../stuff/seg_dataset/train/images"
    TRAIN_MASK_DIR = "../stuff/seg_dataset/train/masks"
    VAL_IMG_DIR = "../stuff/seg_dataset/val/images"
    VAL_MASK_DIR = "../stuff/seg_dataset/val/masks"
    
    BATCH_SIZE = 8
    NUM_WORKERS = 0  # Set to 0 on Windows to avoid multiprocessing issues
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    
    # Transforms
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Datasets
    train_dataset = CrackSegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform)
    val_dataset = CrackSegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, val_transform)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = MobileNetV3UNet(out_channels=1, pretrained=True).to(DEVICE)
    
    # Loss and Optimizer
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = eval_one_epoch(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_segmentation_model.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
