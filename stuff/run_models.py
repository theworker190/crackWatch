import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from skimage.morphology import skeletonize
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from segmentation import MobileNetV3UNet


def extract_skeleton_features(mask):
    """
    Extract detailed crack geometry features using skeleton analysis.
    Returns: length_px, avg_width_px, max_width_px, n_branches, skeleton
    """
    H, W = mask.shape
    
    if np.count_nonzero(mask) == 0:
        return 0, 0, 0, 0, np.zeros_like(mask)
    
    # Convert mask to binary (0 or 1)
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Compute skeleton using skimage
    skeleton = skeletonize(binary_mask > 0)
    skeleton = skeleton.astype(np.uint8)
    
    # Length: count skeleton pixels
    length_px = np.count_nonzero(skeleton)
    
    if length_px == 0:
        return 0, 0, 0, 0, skeleton
    
    # Compute distance transform for width measurements (need binary_mask as 0/255)
    binary_mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255
    dist_transform = cv2.distanceTransform(binary_mask_uint8, cv2.DIST_L2, 5)
    
    # Get skeleton pixel coordinates
    skel_ys, skel_xs = np.where(skeleton > 0)
    
    # Width at each skeleton point (distance to nearest boundary)
    widths = []
    for y, x in zip(skel_ys, skel_xs):
        # Distance transform gives distance to nearest zero pixel
        # Width = 2 * distance (radius on both sides)
        width = dist_transform[y, x] * 2
        widths.append(width)
    
    avg_width_px = np.mean(widths) if widths else 0
    max_width_px = np.max(widths) if widths else 0
    
    # Count branch points (skeleton pixels with more than 2 neighbors)
    # Use 3x3 kernel to count neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # Don't count center pixel
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
    # Branch points have 3 or more neighbors
    branch_points = (neighbor_count >= 3) & (skeleton > 0)
    n_branches = np.count_nonzero(branch_points)
    
    return length_px, avg_width_px, max_width_px, n_branches, skeleton


def classify_severity_geometry(length_px, avg_width_px, max_width_px, n_branches, image_width, mask_area_px, max_area, max_length, max_width, max_branches):
    """
    Classify crack severity using weighted formula:
    Severity = α*A_n + β*L_n + γ*W_n + δ*B_n
    
    Where normalized values are calculated as:
    A_n = Area / max_Area
    L_n = Length / max_Length  
    W_n = Width / max_Width
    B_n = Branches / max_Branches
    
    Weights (example): α=0.3, β=0.3, γ=0.3, δ=0.1
    
    Severity ranges:
    0.0-0.3 → Minor
    0.3-0.6 → Moderate
    0.6-1.0 → Severe
    
    Returns: (severity_score, severity_level)
    """
    alpha = 0.3  # area weight
    beta = 0.3   # length weight
    gamma = 0.3  # width weight
    delta = 0.1  # branching weight
    
    # Normalize measurements (avoid division by zero)
    A_n = mask_area_px / max(max_area, 1)
    L_n = length_px / max(max_length, 1)
    W_n = max_width_px / max(max_width, 1)
    B_n = n_branches / max(max_branches, 1)
    
    # Calculate severity score
    severity_score = alpha * A_n + beta * L_n + gamma * W_n + delta * B_n
    
    # Map to severity classes
    if severity_score < 0.3:
        severity_level = "Minor"
    elif severity_score < 0.6:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"
    
    return severity_score, severity_level


def load_segmentation_model(path, device):
    model = MobileNetV3UNet(out_channels=1, pretrained=False)
    model = model.to(device)
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # try if whole checkpoint dict with key 'model'
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
        else:
            raise
    model.eval()
    return model


def build_classification_model(device):
    from torchvision.models import MobileNet_V3_Small_Weights
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    # Replace classifier[3] for binary classification (same as training script)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    
    model = model.to(device)
    return model


def load_classification_weights(model, path, device):
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
        else:
            raise
    model.eval()
    return model


def preprocess_for_classification(pil_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0)


def preprocess_for_segmentation(pil_img):
    # segmentation model was trained with Resize(256,256) + normalization
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0)


def run_on_image(img_path, seg_model, cls_model, device, threshold=0.5, class_threshold=0.5):
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size  # (W,H)
    img_arr = np.array(img)

    # classification
    cls_in = preprocess_for_classification(img).to(device)
    with torch.no_grad():
        cls_out = cls_model(cls_in).squeeze(1)
        prob = torch.sigmoid(cls_out).cpu().item()
    
    has_crack = prob < 0.5
    crack_confidence = 1 - prob  # confidence that it's a crack
    
    # Print result
    print(f"\nImage: {img_path}")
    print(f"Crack confidence: {crack_confidence:.2%}")
    
    overlay_img = None
    
    # Only segment if classification indicates crack (crack_confidence > class_threshold)
    if has_crack and crack_confidence > class_threshold:
        print(f"Result: CRACK DETECTED")
        
        # segmentation
        seg_in = preprocess_for_segmentation(img).to(device)
        with torch.no_grad():
            seg_out = seg_model(seg_in)
            seg_prob = torch.sigmoid(seg_out)[0, 0].cpu().numpy()  # HxW

        # resize mask back to original size
        mask = (seg_prob > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(orig_size, resample=Image.BILINEAR)
        mask_arr = np.array(mask_img)
        
        # Debug: print segmentation stats
        print(f"  Segmentation - min: {seg_prob.min():.4f}, max: {seg_prob.max():.4f}, mean: {seg_prob.mean():.4f}")
        print(f"  Pixels above threshold ({threshold}): {(seg_prob > threshold).sum()} / {seg_prob.size} ({(seg_prob > threshold).sum() / seg_prob.size * 100:.2f}%)")

        # Create overlay: original image with red dots on cracks
        overlay_arr = img_arr.copy()
        crack_pixels = mask_arr > 0
        
        if crack_pixels.sum() == 0:
            print(f"  WARNING: No pixels detected above threshold {threshold}. Try lowering --threshold")
            overlay_img = None
            severity = None
            skeleton_img = None
        else:
            overlay_arr[crack_pixels] = [255, 0, 0]  # Pure red for crack pixels
            print(f"  Marked {crack_pixels.sum()} crack pixels in red")
            
            # Extract detailed geometry features
            length_px, avg_width_px, max_width_px, n_branches, skeleton = extract_skeleton_features(mask_arr)
            
            # Calculate mask area
            mask_area_px = crack_pixels.sum()
            
            max_area = orig_size[0] * orig_size[1] * 0.5  
            max_length = orig_size[0] * 1.5  
            max_width = 50  
            max_branches = 20  
            
            severity_score, severity = classify_severity_geometry(
                length_px, avg_width_px, max_width_px, n_branches, 
                orig_size[0], mask_area_px, max_area, max_length, max_width, max_branches
            )
            
            skeleton_arr = img_arr.copy()
            skeleton_pixels = skeleton > 0
            skeleton_arr[skeleton_pixels] = [255, 255, 0]  # Yellow for skeleton
            
            # Mark branch points in green if any
            if n_branches > 0:
                kernel = np.ones((3, 3), dtype=np.uint8)
                kernel[1, 1] = 0
                neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
                branch_points = (neighbor_count >= 3) & (skeleton > 0)
                skeleton_arr[branch_points] = [0, 255, 0]  # Green for branches
            
            skeleton_img = skeleton_arr
            
            print(f"\n  Crack Geometry Analysis:")
            print(f"    Area: {mask_area_px} pixels")
            print(f"    Length: {length_px} pixels ({length_px/orig_size[0]:.2f}x image width)")
            print(f"    Average width: {avg_width_px:.2f} pixels")
            print(f"    Maximum width: {max_width_px:.2f} pixels")
            print(f"    Branch points: {n_branches}")
            print(f"    Severity Score: {severity_score:.3f}")
            print(f"    Severity Level: {severity}")
        
        overlay_img = overlay_arr
    else:
        print(f"Result: NO CRACK")
        overlay_img = None
        skeleton_img = None
        severity = None

    return has_crack, crack_confidence, overlay_img, skeleton_img, img_arr, severity


def main():
    parser = argparse.ArgumentParser(description='Run segmentation and classification on images')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--seg-model', default='best_segmentation_model.pth', help='Path to seg model')
    parser.add_argument('--class-model', default='best_classification_model.pth', help='Path to class model')
    parser.add_argument('--device', default=None, help='Device: cpu or cuda (auto if omitted)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Mask threshold (segmentation probability cutoff)')
    parser.add_argument('--class-threshold', type=float, default=0.5, help='Min classification prob to run segmentation')

    args = parser.parse_args()

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # load models
    print('Loading models...')
    seg_path = Path(args.seg_model)
    cls_path = Path(args.class_model)

    if not seg_path.exists():
        seg_path = SCRIPT_DIR / seg_path
    if not cls_path.exists():
        cls_path = SCRIPT_DIR / cls_path

    if not seg_path.exists() or not cls_path.exists():
        raise FileNotFoundError('Model file(s) not found. Checked: ' + str(seg_path) + ' and ' + str(cls_path))

    seg_model = load_segmentation_model(str(seg_path), device)
    cls_model = build_classification_model(device)
    cls_model = load_classification_weights(cls_model, str(cls_path), device)

    # Process single image
    img_path = args.input
    if not Path(img_path).exists():
        print(f'Image not found: {img_path}')
        return

    result = run_on_image(
        img_path, seg_model, cls_model, device, 
        threshold=args.threshold, class_threshold=args.class_threshold
    )
    has_crack, crack_confidence, overlay_img, skeleton_img, original_img, severity = result

    # Display result
    if has_crack and overlay_img is not None:
        # Create 3-panel visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(overlay_img)
        axes[1].set_title('Mask Overlay (Red)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(skeleton_img)
        title = 'Skeleton (Yellow)'
        if severity:
            if isinstance(severity, tuple):
                score, level = severity
                title += f'\n\nSeverity: {level} ({score:.3f})'
            else:
                title += f'\n\nSeverity: {severity}'
        axes[2].set_title(title, fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Crack Analysis - Confidence: {crack_confidence:.2%}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        # Just show original
        plt.figure(figsize=(8, 6))
        plt.imshow(original_img)
        plt.title(f'No Crack Detected\n(Crack Confidence: {crack_confidence:.2%})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
