"""
Train only ResNet50 for Classification
Uses existing MobileNet results if available

FAIR COMPARISON SETUP:
- Batch size: 32 (same as MobileNet baseline)
- Learning rate: 1e-4 (same as MobileNet baseline)
- Epochs: 20 (same as MobileNet baseline)
- Optimizer: Adam (same as MobileNet baseline)
- Augmentations: Rotation(20), HFlip, ColorJitter (same as MobileNet baseline)
- Image size: 224x224 (standard for both models)
- Loss: BCEWithLogitsLoss with pos_weight (same as MobileNet baseline)
"""
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import json
import time
import os


# Transform for training (with augmentation) - SAME AS MOBILENET BASELINE
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Transform for validation/test (NO augmentation) - SAME AS MOBILENET BASELINE
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def prepare_dataloaders(dataset_path, batch_size=32, num_workers=0):
    """Prepare train/val/test dataloaders with consistent splits"""
    base_dataset = ImageFolder(dataset_path)
    
    train_len = int(0.7 * len(base_dataset))
    val_len = int(0.15 * len(base_dataset))
    test_len = len(base_dataset) - train_len - val_len
    
    train_indices, val_indices, test_indices = random_split(
        range(len(base_dataset)),
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_ds = Subset(ImageFolder(dataset_path, transform=train_transform), train_indices.indices)
    val_ds = Subset(ImageFolder(dataset_path, transform=test_transform), val_indices.indices)
    test_ds = Subset(ImageFolder(dataset_path, transform=test_transform), test_indices.indices)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    class_counts = Counter(base_dataset.targets)
    num_class0 = class_counts[0]
    num_class1 = class_counts[1]
    
    print(f"Dataset split:")
    print(f"  Train: {train_len}, Val: {val_len}, Test: {test_len}")
    print(f"Class distribution:")
    print(f"  crack (class 0):     {num_class0}")
    print(f"  no_crack (class 1):  {num_class1}")
    
    return train_loader, val_loader, test_loader, num_class0, num_class1


def create_resnet50_model(pretrained=True, device='cuda'):
    """Create ResNet50 classification model"""
    model = resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0
    total_batches = len(loader)
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # Progress indicator every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            print(f"  Batch [{batch_idx+1}/{total_batches}] - Loss: {loss.item():.4f}", flush=True)
        elif (batch_idx + 1) % 10 == 0:
            print(".", end="", flush=True)
    
    return running_loss / len(loader)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    
    return running_loss / len(loader)


def evaluate_model_detailed(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'crack': {
                'precision': float(precision_per_class[0]),
                'recall': float(recall_per_class[0]),
                'f1_score': float(f1_per_class[0])
            },
            'no_crack': {
                'precision': float(precision_per_class[1]),
                'recall': float(recall_per_class[1]),
                'f1_score': float(f1_per_class[1])
            }
        }
    }
    
    return metrics


def main():
    DATASET_PATH = "classification_dataset"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 20
    BATCH_SIZE = 64  # Increased for RTX 5090 (faster training)
    LEARNING_RATE = 1e-4
    
    print("="*60)
    print("Training ResNet50 Classification Model")
    print("="*60)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}\n")
    
    # Prepare data
    train_loader, val_loader, test_loader, num_class0, num_class1 = prepare_dataloaders(
        DATASET_PATH, batch_size=BATCH_SIZE, num_workers=4  # Parallel data loading for speed
    )
    
    # Create model
    model = create_resnet50_model(pretrained=True, device=DEVICE)
    
    # Loss function with class weights
    pos_weight = torch.tensor([num_class0 / num_class1], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
    
    # Resume from checkpoint if exists
    checkpoint_path = 'checkpoint_resnet50_classification.pth'
    if os.path.exists(checkpoint_path):
        print(f"\nFound checkpoint! Resuming training...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        training_history = checkpoint['training_history']
        print(f"Resuming from epoch {start_epoch + 1}\n")
    
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
            torch.save(model.state_dict(), 'best_resnet50_classification.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})", flush=True)
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print(f"\nResNet50 - Test Set Evaluation:")
    print("-" * 60)
    metrics = evaluate_model_detailed(model, test_loader, DEVICE)
    
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"  Crack (0):    P={metrics['per_class']['crack']['precision']:.4f}, "
          f"R={metrics['per_class']['crack']['recall']:.4f}, "
          f"F1={metrics['per_class']['crack']['f1_score']:.4f}")
    print(f"  No Crack (1): P={metrics['per_class']['no_crack']['precision']:.4f}, "
          f"R={metrics['per_class']['no_crack']['recall']:.4f}, "
          f"F1={metrics['per_class']['no_crack']['f1_score']:.4f}")
    print(f"\nTraining time: {training_time:.2f}s")
    
    metrics['training_time'] = training_time
    metrics['training_history'] = training_history
    metrics['model_name'] = 'resnet50'
    
    # Load existing MobileNet results if available
    comparison = {'resnet50': metrics}
    
    if os.path.exists('classification_comparison_results.json'):
        with open('classification_comparison_results.json', 'r') as f:
            existing = json.load(f)
            if 'mobilenet' in existing:
                comparison['mobilenet'] = existing['mobilenet']
                print("\n✓ Loaded existing MobileNet results")
    
    # Save results
    with open('classification_comparison_results.json', 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print("\n✓ Results saved to 'classification_comparison_results.json'")
    print("✓ Model saved to 'best_resnet50_classification.pth'")


if __name__ == "__main__":
    main()
