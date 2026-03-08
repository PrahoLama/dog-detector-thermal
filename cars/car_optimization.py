#!/usr/bin/env python3
"""
Car Detection Optimization via Blue Channel Thermal Enhancement
================================================================
Goal: Apply proven dog formula to cars (1648 instances)

Strategy: Use exact winning dog approach (74.9% mAP50)
  - Blue channel thermal @ 640px
  - 3x oversampling
  - YOLOv8n, 100 epochs
  - Default augmentation

Dataset: 1648 car instances in 795 images (4x more than dogs)
Expected: Similar or better performance than dogs (74.9%)

Usage:
  python3 car_optimization.py prepare_oversample_3x
  python3 car_optimization.py train_oversample_3x
  python3 car_optimization.py evaluate_oversample_3x
"""

import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import random

# ═══════════════════════════ CONFIG ═══════════════════════════

BASE = Path("/home/praho/Documents/Job/BlajADER")
FUSED = BASE / "fused_dataset"
EXP = BASE / "car_experiment"
EXP.mkdir(exist_ok=True)

# Original class IDs in fused_dataset: 2=car, 4=dog, 7=person
CAR_CLASS_ID = 2
TARGET_CLASS = 0  # Remap car to class 0 for binary detector

# Proven winning configuration from dogs
VARIANTS = {
    "oversample_3x": {
        "desc": "Blue channel, YOLOv8n, 640px, 3x oversampling - PROVEN DOG FORMULA",
        "model": "yolov8n",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "augmentation": "default",
        "preprocessing": "blue",
        "oversample": 3,
    }
}

# ═══════════════════════════ HELPERS ═══════════════════════════

def blue_channel_extraction(img):
    """Extract blue channel (highest thermal info)"""
    if len(img.shape) == 2:
        return img
    return img[:, :, 0]  # OpenCV is BGR

def extract_car_subset():
    """Extract all images with car annotations"""
    src_imgs = FUSED / "images"
    src_labels = FUSED / "labels"
    
    car_images = []
    car_labels = []
    
    print("📂 Scanning for car annotations...")
    for label_file in tqdm(list(src_labels.glob("*.txt"))):
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        # Filter for car class (2)
        car_lines = [line for line in lines if line.split()[0] == str(CAR_CLASS_ID)]
        
        if car_lines:
            img_file = src_imgs / f"{label_file.stem}.jpg"
            if img_file.exists():
                car_images.append(img_file)
                # Remap to class 0
                remapped_lines = [f"0 {' '.join(line.split()[1:])}" for line in car_lines]
                car_labels.append(remapped_lines)
    
    print(f"✅ Found {len(car_images)} images with {sum(len(labels) for labels in car_labels)} car instances")
    return car_images, car_labels

def split_dataset(images, labels, train_ratio=0.75, val_ratio=0.15, seed=42):
    """Split into train/val/test"""
    random.seed(seed)
    indices = list(range(len(images)))
    random.shuffle(indices)
    
    n_train = int(len(images) * train_ratio)
    n_val = int(len(images) * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return {
        'train': ([images[i] for i in train_idx], [labels[i] for i in train_idx]),
        'val': ([images[i] for i in val_idx], [labels[i] for i in val_idx]),
        'test': ([images[i] for i in test_idx], [labels[i] for i in test_idx]),
    }

def prepare_variant(variant_name, config):
    """Prepare dataset for specific variant"""
    variant_dir = EXP / variant_name
    variant_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🔧 Preparing variant: {variant_name}")
    print(f"📋 {config['desc']}")
    print(f"{'='*60}\n")
    
    # Extract car subset
    car_images, car_labels = extract_car_subset()
    
    # Split dataset
    splits = split_dataset(car_images, car_labels)
    
    # Process each split
    for split_name, (imgs, labels) in splits.items():
        print(f"\n📊 Processing {split_name} split ({len(imgs)} images)...")
        
        split_dir = variant_dir / f"dataset/{split_name}"
        img_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle oversampling for train split
        oversample = config.get('oversample', 1)
        if split_name == 'train' and oversample > 1:
            print(f"  🔄 Applying {oversample}x oversampling to training set...")
            orig_count = len(imgs)
            imgs = imgs * oversample
            labels = labels * oversample
            print(f"  ✅ Expanded to {len(imgs)} training images ({orig_count} × {oversample})")
        
        # Copy and preprocess images
        for idx, (img_path, label_lines) in enumerate(tqdm(list(zip(imgs, labels)), desc=f"  {split_name}")):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Apply preprocessing
            if config.get('preprocessing') == 'blue':
                img = blue_channel_extraction(img)
                # Convert to 3-channel for YOLO
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Add suffix for oversampled copies to avoid overwriting
            if split_name == 'train' and oversample > 1:
                copy_idx = idx % oversample
                suffix = f"_copy{copy_idx}" if copy_idx > 0 else ""
            else:
                suffix = ""
            
            # Save image
            dst_img = img_dir / f"{img_path.stem}{suffix}.jpg"
            cv2.imwrite(str(dst_img), img)
            
            # Save label
            dst_label = label_dir / f"{img_path.stem}{suffix}.txt"
            with open(dst_label, 'w') as f:
                f.write('\n'.join(label_lines) + '\n')
    
    # Create dataset.yaml
    yaml_content = f"""# Car detection dataset - {variant_name}
path: {variant_dir / 'dataset'}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['car']
"""
    
    with open(variant_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    # Print summary
    train_count = len(list((variant_dir / "dataset/train/labels").glob("*.txt")))
    val_count = len(list((variant_dir / "dataset/val/labels").glob("*.txt")))
    test_count = len(list((variant_dir / "dataset/test/labels").glob("*.txt")))
    
    print(f"\n✅ Dataset prepared:")
    print(f"   Train: {train_count} images")
    print(f"   Val:   {val_count} images")
    print(f"   Test:  {test_count} images")
    print(f"   Config: {variant_dir / 'dataset.yaml'}")

def train_variant(variant_name, config):
    """Train specific variant"""
    from ultralytics import YOLO
    
    variant_dir = EXP / variant_name
    yaml_path = variant_dir / "dataset.yaml"
    
    if not yaml_path.exists():
        print(f"❌ Dataset not prepared. Run: python3 {sys.argv[0]} prepare_{variant_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"🚀 Training variant: {variant_name}")
    print(f"📋 {config['desc']}")
    print(f"{'='*60}\n")
    
    # Check if training already completed
    results_csv = variant_dir / "runs/train/results.csv"
    if results_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            if len(df) >= config['epochs']:
                print(f"✅ Training already completed ({len(df)} epochs)")
                print(f"📊 Resuming evaluation only...")
                # Just evaluate
                model = YOLO(variant_dir / "runs/train/weights/best.pt")
                results = model.val(data=str(yaml_path), split='test')
                print(f"\n🏆 Test mAP50: {results.box.map50*100:.1f}%")
                return
        except Exception as e:
            print(f"⚠️ Could not check completion status: {e}")
    
    # Initialize model
    model_name = config['model']
    model_path = BASE / f"{model_name}.pt"
    
    if not model_path.exists():
        print(f"📥 Downloading {model_name}...")
        model = YOLO(model_name)
    else:
        print(f"📦 Loading {model_name} from {model_path}")
        model = YOLO(str(model_path))
    
    # Train
    print(f"\n🏋️ Starting training...")
    print(f"  Model: {model_name}")
    print(f"  Image size: {config['imgsz']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch: {config['batch']}")
    
    results = model.train(
        data=str(yaml_path),
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        patience=20,
        save=True,
        project=str(variant_dir / "runs"),
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        seed=42,
        deterministic=True,
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        workers=2,  # Prevent process explosion
    )
    
    print(f"\n✅ Training completed!")
    print(f"📂 Weights: {variant_dir / 'runs/train/weights/best.pt'}")
    
    # Evaluate on test set
    print(f"\n📊 Running final evaluation on test set...")
    best_model = YOLO(variant_dir / "runs/train/weights/best.pt")
    test_results = best_model.val(data=str(yaml_path), split='test')
    
    print(f"\n{'='*60}")
    print(f"🏆 FINAL RESULTS - {variant_name}")
    print(f"{'='*60}")
    print(f"Test mAP50:       {test_results.box.map50*100:.1f}%")
    print(f"Test mAP50-95:    {test_results.box.map*100:.1f}%")
    print(f"Test Precision:   {test_results.box.p[0]*100:.1f}%")
    print(f"Test Recall:      {test_results.box.r[0]*100:.1f}%")
    print(f"{'='*60}\n")

def evaluate_variant(variant_name):
    """Evaluate trained variant"""
    from ultralytics import YOLO
    
    variant_dir = EXP / variant_name
    model_path = variant_dir / "runs/train/weights/best.pt"
    yaml_path = variant_dir / "dataset.yaml"
    
    if not model_path.exists():
        print(f"❌ Model not trained. Run: python3 {sys.argv[0]} train_{variant_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluating: {variant_name}")
    print(f"{'='*60}\n")
    
    model = YOLO(model_path)
    
    # Test set evaluation
    print("Running test set evaluation...")
    test_results = model.val(data=str(yaml_path), split='test')
    
    print(f"\n{'='*60}")
    print(f"🏆 TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"mAP50:       {test_results.box.map50*100:.1f}%")
    print(f"mAP50-95:    {test_results.box.map*100:.1f}%")
    print(f"Precision:   {test_results.box.p[0]*100:.1f}%")
    print(f"Recall:      {test_results.box.r[0]*100:.1f}%")
    print(f"{'='*60}\n")

# ═══════════════════════════ MAIN ═══════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print(f"  python3 {sys.argv[0]} prepare_oversample_3x")
        print(f"  python3 {sys.argv[0]} train_oversample_3x")
        print(f"  python3 {sys.argv[0]} evaluate_oversample_3x")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "prepare_oversample_3x":
        prepare_variant("oversample_3x", VARIANTS["oversample_3x"])
    
    elif command == "train_oversample_3x":
        train_variant("oversample_3x", VARIANTS["oversample_3x"])
    
    elif command == "evaluate_oversample_3x":
        evaluate_variant("oversample_3x")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("\nAvailable commands:")
        print("  prepare_oversample_3x")
        print("  train_oversample_3x")
        print("  evaluate_oversample_3x")
        sys.exit(1)
