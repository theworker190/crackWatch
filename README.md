# CrackWatch: AI-Powered Crack Detection & Severity Analysis

A comprehensive computer vision system for detecting and analyzing structural cracks using deep learning. Features both classification and segmentation models with real-time severity assessment.

---

## Overview

CrackWatch combines deep learning models to:
- **Classify** images as cracked or non-cracked (binary classification)
- **Segment** crack regions with pixel-level precision
- **Analyze** crack severity using geometric features (area, length, width, branching)
- **Deploy** lightweight models suitable for edge devices (MobileNetV3)

### Key Features
- ✅ Dual-model approach: ResNet50 & MobileNetV3 (16x smaller, edge-ready)
- ✅ Transfer learning from ImageNet classification to crack segmentation
- ✅ Weighted severity formula: `Severity = 0.3·Area + 0.3·Length + 0.3·Width + 0.1·Branching`
- ✅ Manual annotation support with transparent PNG masks
- ✅ Real-time inference with red overlay visualization
- ✅ Comprehensive evaluation with confusion matrices

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/theworker190/crackWatch.git
cd crackWatch

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference

Detect cracks in a single image:

```bash
cd stuff
python run_models.py --input ../local_dataset/crack/crack1_10.jpeg --threshold 0.15
```

**Arguments:**
- `--input`: Path to input image
- `--threshold`: Segmentation probability threshold (default: 0.3, lower = more sensitive)
- `--class-threshold`: Min confidence to run segmentation (default: 0.5)
- `--seg-model`: Path to segmentation model (default: `best_segmentation_model.pth`)
- `--class-model`: Path to classification model (default: `best_classification_model.pth`)

**Output:**
- 3-panel visualization: Original | Red Overlay | Skeleton Analysis
- Crack geometry metrics: area, length, width, branching
- Severity classification: Minor / Moderate / Severe

##  Training

### Train Classification Model

```bash
cd stuff
python classification.py
```

**Dataset:** SDNET2018 (40,000+ images)
- Train: 70% | Val: 15% | Test: 15%
- Augmentation: RandomRotation, HorizontalFlip, ColorJitter
- Loss: BCEWithLogitsLoss (class-weighted)
- Optimizer: Adam (lr=1e-4)

### Train Segmentation Model

```bash
cd stuff
python segmentation.py
```

**Dataset:** CrackForest / DeepCrack / Local Annotations
- Train/Val split: 80/20
- Augmentation: HorizontalFlip, Rotate(±20°), RandomBrightnessContrast
- Loss: BCE + Dice (combined)
- Optimizer: Adam (lr=1e-4)

**Mask Format:**
- RGBA transparent PNG
- Black (RGB≈0) where crack is drawn
- Opaque (alpha=255) for drawn regions
- Transparent elsewhere

---

## 🔬 Severity Analysis

Cracks are classified using a weighted geometric formula:

```
Severity = α·A_n + β·L_n + γ·W_n + δ·B_n
```

**Where:**
- `A_n` = Normalized area (crack pixels / max area)
- `L_n` = Normalized length (skeleton pixels / max length)
- `W_n` = Normalized width (max width / threshold)
- `B_n` = Normalized branching (branch points / max branches)
- Weights: α=0.3, β=0.3, γ=0.3, δ=0.1

**Severity Levels:**
- **Minor:** 0.0 - 0.3 (small surface cracks)
- **Moderate:** 0.3 - 0.6 (visible cracks requiring monitoring)
- **Severe:** 0.6 - 1.0 (critical structural damage)

---

## 🗂️ Datasets

### 1. SDNET2018 (Classification)
- **Location:** `classification_dataset/` (preprocessed)
- **Original:** `archive22222/` (Decks/Pavements/Walls)
- **Samples:** 40,000+ concrete surface images
- **Classes:** crack / no_crack

### 2. CrackForest (Segmentation)
- **Location:** `stuff/crackforest/`
- **Samples:** 118 images with .mat ground truth
- **Type:** Pavement crack segmentation

### 3. DeepCrack (Segmentation)
- **Location:** `stuff/DeepCrack/`
- **Samples:** 537 training, 200 test
- **Type:** High-resolution crack segmentation

### 4. Local Annotations (Manual)
- **Location:** `local_dataset/`
- **Samples:** 19 transparent PNG masks
- **Type:** Hand-drawn crack annotations

---

## 📝 Model Architecture

### MobileNetV3-UNet
- **Encoder:** Pretrained MobileNetV3-Small (ImageNet)
- **Decoder:** 4-level U-Net with skip connections
- **Output:** Single-channel crack probability map
- **Size:** 2.1M parameters (edge-deployable)

### ResNet50-UNet
- **Encoder:** Pretrained ResNet50 (ImageNet)
- **Decoder:** 5-level U-Net with skip connections
- **Output:** Single-channel crack probability map
- **Size:** 34.2M parameters (high accuracy)


## 📄 License

MIT License - See LICENSE file for details

---

## Authors
<h1>Members of Group 6</h1></br>
<h3>Abdelrahman Feteha B00094970</h3></br>
<h3>Ahmad Elsahfie     B00095145</h3></br>
<h3>Ahmed Mehaisi      B00094989</h3></br>

---

##  Acknowledgments

- **SDNET2018** dataset by Utah State University https://digitalcommons.usu.edu/all_datasets/48/
- **CrackForest** dataset by Shi et al. https://github.com/cuilimeng/CrackForest-dataset
- **DeepCrack** dataset by Liu et al. https://github.com/yhlleo/DeepCrack
---
