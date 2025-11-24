"""
Train only ResNet50-UNet for Segmentation
Uses existing MobileNet-UNet results if available

FAIR COMPARISON SETUP:
- Batch size: 8 (same as MobileNet-UNet baseline)
- Learning rate: 1e-4 (same as MobileNet-UNet baseline)
- Epochs: 20 (same as MobileNet-UNet baseline)
- Optimizer: Adam (same as MobileNet-UNet baseline)
- Augmentations: HFlip(0.5), Rotate(20, 0.5), BrightnessContrast(0.3) (same as baseline)
- Image size: 256x256 (same as MobileNet-UNet baseline)
- Loss: BCEDiceLoss (same as MobileNet-UNet baseline)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet50
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import time


class CrackSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, invert_mask=False):
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
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        if self.invert_mask:
            mask = 255 - mask
        
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask


class ResNet50UNet(nn.Module):
    def __init__(self, out_channels=1, pretrained=True):
        super(ResNet50UNet, self).__init__()
        
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights)
        
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        
        self.bottleneck = resnet.layer4
        
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec4 = self.conv_block(1024 + 1024, 1024)
        
        self.upconv3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec3 = self.conv_block(512 + 512, 512)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = self.conv_block(256 + 256, 256)
        
        self.upconv1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.dec1 = self.conv_block(64 + 64, 64)
        
        self.final_up = nn.ConvTranspose2d(64, 64, 2, stride=2)
        
        self.out = nn.Conv2d(64, out_channels, 1)
    
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
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        bottleneck = self.bottleneck(enc4)
        
        dec4 = self.upconv4(bottleneck)
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
        
        final = self.final_up(dec1)
        out = self.out(final)
        
        if out.shape[2:] != x.shape[2:]:
            out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out


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


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0
    total_batches = len(loader)
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.unsqueeze(1).to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # Progress indicator every 50 batches (fewer batches in segmentation)
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            print(f"  Batch [{batch_idx+1}/{total_batches}] - Loss: {loss.item():.4f}", flush=True)
    
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
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()


def calculate_dice(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-8) / (pred.sum() + target.sum() + 1e-8)
    
    return dice.item()


def evaluate_segmentation_detailed(model, loader, device, threshold=0.5):
    model.eval()
    
    all_ious = []
    all_dices = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)
            
            outputs = model(images)
            
            batch_iou = calculate_iou(outputs, masks, threshold)
            batch_dice = calculate_dice(outputs, masks, threshold)
            
            all_ious.append(batch_iou)
            all_dices.append(batch_dice)
            
            pred_flat = (torch.sigmoid(outputs) > threshold).cpu().numpy().flatten()
            target_flat = masks.cpu().numpy().flatten()
            
            all_preds.extend(pred_flat)
            all_targets.extend(target_flat)
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0
    )
    
    mean_iou = np.mean(all_ious)
    mean_dice = np.mean(all_dices)
    
    metrics = {
        'iou': float(mean_iou),
        'dice': float(mean_dice),
        'pixel_accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    return metrics


def main():
    TRAIN_IMG_DIR = "stuff/seg_dataset/train/images"
    TRAIN_MASK_DIR = "stuff/seg_dataset/train/masks"
    VAL_IMG_DIR = "stuff/seg_dataset/val/images"
    VAL_MASK_DIR = "stuff/seg_dataset/val/masks"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 20  # Same as MobileNet baseline
    BATCH_SIZE = 16  # Increased for RTX 5090 (faster training)
    LEARNING_RATE = 1e-4  # Same as MobileNet baseline
    NUM_WORKERS = 4  # Parallel data loading for speed
    
    print("="*60)
    print("Training ResNet50-UNet Segmentation Model")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Num Workers: {NUM_WORKERS}\n")
    
    # Transforms - SAME AS MOBILENET BASELINE
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
    train_dataset = CrackSegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform, invert_mask=False)
    val_dataset = CrackSegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, val_transform, invert_mask=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}\n")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = ResNet50UNet(out_channels=1, pretrained=True).to(DEVICE)
    
    # Loss and Optimizer
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Mixed precision training for speed (RTX 5090 support)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    best_val_loss = float('inf')
    training_history = []
    start_epoch = 0
    
    # Load best model if exists to continue training
    best_model_path = 'best_resnet50_segmentation.pth'
    checkpoint_path = 'checkpoint_resnet50_segmentation.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"\nFound checkpoint! Resuming training...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        training_history = checkpoint['training_history']
        print(f"Resuming from epoch {start_epoch + 1}\n")
    elif os.path.exists(best_model_path):
        print(f"\nFound best model! Loading to continue training...")
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model, starting fresh from epoch 1\n")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Starting...", flush=True)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Training complete. Evaluating...", flush=True)
        val_loss = eval_one_epoch(model, val_loader, criterion, DEVICE)
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'training_history': training_history
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint saved", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_resnet50_segmentation.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})", flush=True)
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print(f"\nResNet50-UNet - Validation Set Evaluation:")
    print("-" * 60)
    metrics = evaluate_segmentation_detailed(model, val_loader, DEVICE)
    
    print(f"IoU:             {metrics['iou']:.4f}")
    print(f"Dice:            {metrics['dice']:.4f}")
    print(f"Pixel Accuracy:  {metrics['pixel_accuracy']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"F1 Score:        {metrics['f1_score']:.4f}")
    print(f"\nTraining time: {training_time:.2f}s")
    
    metrics['training_time'] = training_time
    metrics['training_history'] = training_history
    metrics['model_name'] = 'resnet50'
    
    # Load existing MobileNet results if available
    comparison = {'resnet50': metrics}
    
    if os.path.exists('segmentation_comparison_results.json'):
        with open('segmentation_comparison_results.json', 'r') as f:
            existing = json.load(f)
            if 'mobilenet' in existing:
                comparison['mobilenet'] = existing['mobilenet']
                print("\n✓ Loaded existing MobileNet results")
    
    # Save results
    with open('segmentation_comparison_results.json', 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print("\n✓ Results saved to 'segmentation_comparison_results.json'")
    print("✓ Model saved to 'best_resnet50_segmentation.pth'")


if __name__ == "__main__":
    main()
