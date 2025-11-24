import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import classification_report


# Transform for training (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Transform for validation/test (NO augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 1) Load dataset ONCE with no transform (neutral)
base_dataset = ImageFolder("dataset")

# 2) Split indices only
train_len = int(0.7 * len(base_dataset))
val_len = int(0.15 * len(base_dataset))
test_len = len(base_dataset) - train_len - val_len

train_indices, val_indices, test_indices = random_split(
    range(len(base_dataset)),
    [train_len, val_len, test_len]
)

# 3) Apply transforms separately by wrapping Subset
train_ds = Subset(ImageFolder("dataset", transform=train_transform), train_indices.indices)
val_ds   = Subset(ImageFolder("dataset", transform=test_transform),  val_indices.indices)
test_ds  = Subset(ImageFolder("dataset", transform=test_transform),  test_indices.indices)

train_len, val_len, test_len

batch_size = 32
num_workers = 4   # Adjust based on your CPU

train_loader = DataLoader(
    train_ds, 
    batch_size=batch_size, 
    shuffle=True,          # shuffle only train
    num_workers=num_workers
)

val_loader = DataLoader(
    val_ds, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers
)

test_loader = DataLoader(
    test_ds, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers
)



model = mobilenet_v3_small(pretrained=True)

# Replace last layer for binary classification
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")




# Use the base dataset WITHOUT transforms (fast, labels-only)
base_dataset = ImageFolder("dataset")

# Count occurrences of each class label
class_counts = Counter(base_dataset.targets)

# Extract counts
num_class0 = class_counts[0]  # crack
num_class1 = class_counts[1]  # no_crack

# Compute pos_weight for BCEWithLogitsLoss
pos_weight = torch.tensor([num_class0 / num_class1], 
                          device="cuda" if torch.cuda.is_available() else "cpu")

print("Class distribution:")
print(f"  crack (class 0):     {num_class0}")
print(f"  no_crack (class 1):  {num_class1}")
print(f"\nComputed pos_weight = {pos_weight.item():.4f}")


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)  # float for BCE

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

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

device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 20
best_val_loss = float('inf')

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = eval_one_epoch(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_classification_model.pth')
        print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)

            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=['crack', 'no_crack']))

# Evaluate on test set
print("\n" + "="*50)
print("Final Test Set Evaluation")
print("="*50)
evaluate_model(model, test_loader, device)
