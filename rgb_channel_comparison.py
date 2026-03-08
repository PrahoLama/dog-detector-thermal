#!/usr/bin/env python3
"""
RGB Channel Comparison - Non-Thermal Images
Tests R, G, B channel separation + RGB full for dogs, cars, persons
Source: fused_dataset (regular RGB drone images, NOT thermal)
"""

import os
import sys
import time
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import yaml
import cv2
from tqdm import tqdm
from collections import defaultdict

BASE_DIR = Path("/home/praho/Documents/Job/BlajADER")
FUSED_DATASET = BASE_DIR / "fused_dataset"
OUTPUT_BASE = BASE_DIR / "rgb_channel_experiment"

# Global tracking
current_model = ""
current_epoch = 0
total_epochs = 0
best_map50 = 0.0

def print_loading_bar(epoch, total, map50, precision, recall, best):
    """Print a loading bar style progress"""
    percentage = (epoch / total) * 100
    filled = int(percentage / 2.5)  # 40 char bar
    bar = "█" * filled + "░" * (40 - filled)
    
    star = "⭐" if map50 >= best else ""
    
    print(f"\r{current_model} [{bar}] {percentage:5.1f}% | "
          f"Epoch {epoch}/{total} | mAP50: {map50:5.1f}% {star} | "
          f"P: {precision:5.1f}% R: {recall:5.1f}%", 
          end="", flush=True)

def on_train_epoch_end(trainer):
    """Callback for epoch progress"""
    global best_map50, current_epoch
    current_epoch = trainer.epoch + 1
    total = trainer.epochs
    
    metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
    map50 = metrics.get('metrics/mAP50(B)', 0) * 100 if metrics else 0
    precision = metrics.get('metrics/precision(B)', 0) * 100 if metrics else 0
    recall = metrics.get('metrics/recall(B)', 0) * 100 if metrics else 0
    
    if map50 > best_map50:
        best_map50 = map50
    
    print_loading_bar(current_epoch, total, map50, precision, recall, best_map50)

def split_dataset_by_class(images_dir, labels_dir, class_id, class_name):
    """
    Extract images containing a specific class and split train/val/test
    
    Args:
        images_dir: Path to fused_dataset/images
        labels_dir: Path to fused_dataset/labels
        class_id: YOLO class ID (0=backpack, 2=car, 4=dog, 7=person, etc.)
        class_name: Name for output ('dog', 'car', 'person')
    
    Returns:
        List of (image_path, label_path) tuples
    """
    print(f"\n  Extracting {class_name} images...")
    
    # Map class names to IDs from classes.txt
    class_map = {
        'dog': 4,
        'car': 2,
        'person': 7
    }
    
    target_class_id = class_map[class_name]
    
    matching_pairs = []
    
    for label_file in labels_dir.glob("*.txt"):
        # Read label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Check if this label contains our target class
        has_target = False
        for line in lines:
            if line.strip():
                class_id_in_line = int(line.split()[0])
                if class_id_in_line == target_class_id:
                    has_target = True
                    break
        
        if has_target:
            # Find corresponding image
            img_name = label_file.stem + ".jpg"
            img_path = images_dir / img_name
            
            if img_path.exists():
                matching_pairs.append((img_path, label_file))
    
    print(f"  Found {len(matching_pairs)} images with {class_name}")
    return matching_pairs

def oversample_dataset(pairs, target_multiplier=3):
    """Oversample by duplicating image-label pairs"""
    print(f"  Oversampling {len(pairs)} pairs by {target_multiplier}x...")
    
    oversampled = []
    for img_path, label_path in pairs:
        for i in range(target_multiplier):
            oversampled.append((img_path, label_path, i))
    
    print(f"  After oversampling: {len(oversampled)} pairs")
    return oversampled

def extract_channel(img_path, method):
    """
    Extract channel from RGB image
    method: 'rgb_full', 'red', 'green', 'blue'
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    if method == 'rgb_full':
        return img
    elif method == 'red':
        red = img[:, :, 2]  # BGR -> Red is index 2
        return cv2.merge([red, red, red])
    elif method == 'green':
        green = img[:, :, 1]
        return cv2.merge([green, green, green])
    elif method == 'blue':
        blue = img[:, :, 0]
        return cv2.merge([blue, blue, blue])
    else:
        raise ValueError(f"Unknown method: {method}")

def create_class_dataset(class_name, method, pairs):
    """
    Create a YOLO dataset for specific class and channel method
    
    Args:
        class_name: 'dog', 'car', or 'person'
        method: 'rgb_full', 'red', 'green', 'blue'
        pairs: List of (img_path, label_path, copy_num) tuples
    
    Returns:
        Path to created dataset
    """
    print(f"\n{'='*80}")
    print(f"📦 CREATING {class_name.upper()} - {method.upper()} DATASET")
    print(f"{'='*80}")
    
    output_dir = OUTPUT_BASE / f"{class_name}_{method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map class names to IDs
    class_map = {'dog': 4, 'car': 2, 'person': 7}
    target_class_id = class_map[class_name]
    
    # Shuffle for train/val/test split
    random.shuffle(pairs)
    
    # Split: 80% train, 10% val, 10% test
    n_total = len(pairs)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    splits = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train+n_val],
        'test': pairs[n_train+n_val:]
    }
    
    # Process each split
    for split_name, split_pairs in splits.items():
        if not split_pairs:
            continue
        
        images_out = output_dir / 'images' / split_name
        labels_out = output_dir / 'labels' / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Processing {split_name}: {len(split_pairs)} images...")
        
        for img_path, label_path, copy_num in tqdm(split_pairs, desc=f"  {split_name}"):
            # Generate unique filename
            base_name = img_path.stem
            if copy_num > 0:
                new_name = f"{base_name}_copy{copy_num}"
            else:
                new_name = base_name
            
            # Process image with channel extraction
            processed_img = extract_channel(img_path, method)
            if processed_img is not None:
                cv2.imwrite(str(images_out / f"{new_name}.jpg"), processed_img)
            
            # Filter and copy label (only keep target class)
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            filtered_lines = []
            for line in lines:
                if line.strip():
                    class_id = int(line.split()[0])
                    if class_id == target_class_id:
                        # Remap to class 0 (single-class detector)
                        parts = line.split()
                        parts[0] = '0'
                        filtered_lines.append(' '.join(parts) + '\n')
            
            if filtered_lines:
                with open(labels_out / f"{new_name}.txt", 'w') as f:
                    f.writelines(filtered_lines)
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': [class_name]
    }
    
    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(f"# {class_name.capitalize()} Detection - RGB {method.upper()}\n")
        f.write(f"# Source: fused_dataset (non-thermal RGB)\n")
        f.write(f"# Method: {method.replace('_', ' ').upper()}\n")
        f.write(f"# Oversampling: 3x\n\n")
        yaml.dump(dataset_yaml, f)
    
    train_imgs = len(list((output_dir / 'images/train').glob("*.jpg")))
    val_imgs = len(list((output_dir / 'images/val').glob("*.jpg")))
    test_imgs = len(list((output_dir / 'images/test').glob("*.jpg")))
    
    print(f"\n✅ Dataset created:")
    print(f"   Train: {train_imgs} images")
    print(f"   Val: {val_imgs} images")
    print(f"   Test: {test_imgs} images")
    
    return output_dir

def train_model(class_name, method, dataset_path):
    """Train YOLOv8n with loading bar progress"""
    global current_model, best_map50, current_epoch
    
    current_model = f"{class_name}_{method}"
    best_map50 = 0.0
    current_epoch = 0
    
    print(f"\n\n{'='*80}")
    print(f"🚀 TRAINING {class_name.upper()} - {method.upper()}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    # Output directory
    output_dir = OUTPUT_BASE / "results"
    
    # Train
    results = model.train(
        data=str(dataset_path / "dataset.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        name=f'{class_name}_{method}',
        project=str(output_dir),
        
        optimizer='AdamW',
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        
        patience=10,
        workers=2,
        device=0,
        seed=42,
        verbose=False,  # Use our custom progress
        save=True,
        plots=True
    )
    
    print()  # New line after progress bar
    elapsed = time.time() - start_time
    
    # Test on test set
    print(f"📊 Testing {class_name} {method}...")
    test_results = model.val(split='test')
    
    metrics = {
        'class': class_name,
        'method': method,
        'time': f"{elapsed/60:.1f} min",
        'map50': float(test_results.box.map50) * 100,
        'map50_95': float(test_results.box.map) * 100,
        'precision': float(test_results.box.p) * 100,
        'recall': float(test_results.box.r) * 100,
    }
    
    print(f"✅ Results: mAP50={metrics['map50']:.1f}% P={metrics['precision']:.1f}% R={metrics['recall']:.1f}%")
    
    return metrics

def main():
    print("="*80)
    print("🎨 RGB CHANNEL COMPARISON - NON-THERMAL IMAGES")
    print("="*80)
    print("\nSource: fused_dataset (regular RGB drone images)")
    print("Testing: R channel, G channel, B channel, RGB full")
    print("Classes: Dogs, Cars, Persons")
    print("Strategy: 3x oversampling per class")
    print(f"\nTotal trainings: 12 (4 methods × 3 classes)")
    print(f"Expected time: ~3-4 hours\n")
    
    OUTPUT_BASE.mkdir(exist_ok=True)
    
    images_dir = FUSED_DATASET / "images"
    labels_dir = FUSED_DATASET / "labels"
    
    # Extract and prepare datasets for each class
    class_datasets = {}
    
    for class_name in ['dog', 'car', 'person']:
        print(f"\n{'#'*80}")
        print(f"# PREPARING {class_name.upper()} DATASETS")
        print(f"{'#'*80}")
        
        # Extract images with this class
        pairs = split_dataset_by_class(images_dir, labels_dir, None, class_name)
        
        # Oversample 3x
        oversampled_pairs = oversample_dataset(pairs, target_multiplier=3)
        
        # Create 4 channel variants
        class_datasets[class_name] = {}
        for method in ['rgb_full', 'red', 'green', 'blue']:
            dataset_path = create_class_dataset(class_name, method, oversampled_pairs.copy())
            class_datasets[class_name][method] = dataset_path
    
    # Train all models
    print(f"\n\n{'#'*80}")
    print(f"# STARTING TRAINING - 12 MODELS")
    print(f"{'#'*80}\n")
    
    all_results = []
    
    for class_name in ['dog', 'car', 'person']:
        for method in ['rgb_full', 'red', 'green', 'blue']:
            try:
                dataset_path = class_datasets[class_name][method]
                metrics = train_model(class_name, method, dataset_path)
                all_results.append(metrics)
            except Exception as e:
                print(f"\n❌ Error training {class_name} {method}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'class': class_name,
                    'method': method,
                    'error': str(e)
                })
    
    # Final summary
    print(f"\n\n{'='*80}")
    print(f"📊 FINAL RESULTS - RGB CHANNEL COMPARISON")
    print(f"{'='*80}\n")
    
    for class_name in ['dog', 'car', 'person']:
        print(f"{'─'*80}")
        print(f"🎯 {class_name.upper()}")
        print(f"{'─'*80}")
        print(f"{'Method':<12} {'mAP50':<10} {'Precision':<12} {'Recall':<10} {'Time'}")
        print(f"{'─'*70}")
        
        best_map = 0
        best_method = None
        
        for r in all_results:
            if r['class'] != class_name:
                continue
            
            if 'error' in r:
                print(f"{r['method']:<12} {'ERROR':<10}")
                continue
            
            if r['map50'] > best_map:
                best_map = r['map50']
                best_method = r['method']
            
            print(f"{r['method']:<12} {r['map50']:>6.1f}%    "
                  f"{r['precision']:>5.1f}%       "
                  f"{r['recall']:>5.1f}%     "
                  f"{r['time']}")
        
        if best_method:
            print(f"\n➡️  Best: {best_method.upper()} ({best_map:.1f}%)\n")
    
    # Save results
    results_file = OUTPUT_BASE / "results_summary.txt"
    with open(results_file, 'w') as f:
        f.write("RGB Channel Comparison Results\n")
        f.write("="*80 + "\n")
        f.write("Source: fused_dataset (non-thermal RGB images)\n\n")
        
        for class_name in ['dog', 'car', 'person']:
            f.write(f"\n{class_name.upper()}:\n")
            f.write("-"*40 + "\n")
            for r in all_results:
                if r['class'] != class_name:
                    continue
                f.write(f"\n{r['method'].upper()}:\n")
                for k, v in r.items():
                    if k != 'class':
                        f.write(f"  {k}: {v}\n")
    
    print(f"\n✅ Results saved to: {results_file}")
    print("\n" + "="*80)
    print("✅ RGB Channel Comparison Complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
