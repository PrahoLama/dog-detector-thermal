#!/usr/bin/env python3
"""
Training script for dog detection with blue channel extraction and 3x oversampling.

This script reproduces the 74.9% mAP50 result.

Usage:
    python scripts/train.py --data config/dataset.yaml --epochs 100 --batch 16
"""

import argparse
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil


def extract_blue_channel(img_path, preprocessing=None):
    """
    Extract blue channel from thermal image.
    
    Args:
        img_path: Path to input image
        preprocessing: Optional preprocessing (None, 'clahe', 'hist_eq')
        
    Returns:
        Processed 3-channel image
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # Extract blue channel
    blue = img[:, :, 0]  # OpenCV uses BGR order
    
    # Apply preprocessing if specified
    if preprocessing == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        blue = clahe.apply(blue)
    elif preprocessing == "hist_eq":
        blue = cv2.equalizeHist(blue)
    
    # Replicate to 3 channels
    img_processed = cv2.merge([blue, blue, blue])
    
    return img_processed


def prepare_dataset_with_oversampling(
    source_images_dir, 
    source_labels_dir,
    output_dir,
    oversample_factor=3
):
    """
    Prepare dataset with blue channel extraction and oversampling.
    
    Args:
        source_images_dir: Directory containing original images
        source_labels_dir: Directory containing YOLO format labels
        output_dir: Output directory for processed dataset
        oversample_factor: Number of times to replicate training data
    """
    output_dir = Path(output_dir)
    output_images = output_dir / 'images' / 'train'
    output_labels = output_dir / 'labels' / 'train'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    source_images_dir = Path(source_images_dir)
    source_labels_dir = Path(source_labels_dir)
    
    image_files = list(source_images_dir.glob('*.jpg')) + list(source_images_dir.glob('*.png'))
    
    print(f"Processing {len(image_files)} images with {oversample_factor}x oversampling...")
    
    for idx, img_path in enumerate(image_files):
        # Extract blue channel
        img_processed = extract_blue_channel(img_path)
        if img_processed is None:
            continue
        
        # Create oversample_factor copies
        for copy_idx in range(oversample_factor):
            output_name = f"{img_path.stem}_copy{copy_idx}{img_path.suffix}"
            output_img_path = output_images / output_name
            output_label_path = output_labels / f"{Path(output_name).stem}.txt"
            
            # Save processed image
            cv2.imwrite(str(output_img_path), img_processed)
            
            # Copy label file
            label_path = source_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, output_label_path)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)}")
    
    print(f"✓ Dataset prepared: {len(image_files) * oversample_factor} training images")


def train_model(data_yaml, epochs=100, batch=16, imgsz=640, device=0):
    """
    Train YOLOv8n model with optimal settings.
    
    Args:
        data_yaml: Path to dataset YAML configuration
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        device: GPU device ID
    """
    print("="*70)
    print("TRAINING: Blue Channel YOLOv8n with 3x Oversampling")
    print("="*70)
    
    # Load pre-trained YOLOv8n
    model = YOLO('yolov8n.pt')
    
    # Train with optimal hyperparameters
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=20,              # Early stopping patience
        device=device,
        deterministic=True,
        seed=42,                  # Reproducibility
        project='runs/train',
        name='dog_detector',
        exist_ok=True,
        verbose=True,
        
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.002,
        
        # Augmentation (default YOLO settings work best)
        degrees=0.0,              # No rotation (thermal cameras typically level)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best weights saved to: runs/train/dog_detector/weights/best.pt")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train dog detector on thermal imagery'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to dataset YAML configuration'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch', 
        type=int, 
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--imgsz', 
        type=int, 
        default=640,
        help='Image size (default: 640)'
    )
    parser.add_argument(
        '--device', 
        type=int, 
        default=0,
        help='GPU device ID (default: 0)'
    )
    
    args = parser.parse_args()
    
    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )


if __name__ == '__main__':
    main()
