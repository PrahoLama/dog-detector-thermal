#!/usr/bin/env python3
"""
Person Detection Optimization via Blue Channel Thermal Enhancement
===================================================================
Goal: Achieve 80%+ AP50 on thermal person detection (baseline unknown)

Strategy based on dog detection success (69.5% → 74.9%):
  1. Baseline: Blue channel thermal @ 640px, YOLOv8n
  2. Light oversampling: 1.5-2x (less aggressive than dogs due to 9x more data)
  3. Larger model: YOLOv8s (11M params - justified with 3661 instances)
  4. Resolution boost: 960px (humans are larger, should benefit more)
  5. Combined best: Merge successful approaches

Why blue channel works for humans:
  - Isolates 36-37°C temperature band (human body heat)
  - Maximum contrast vs background (20-30°C ambient)
  - 3661 instances (9x more than dogs) → expect better performance

Key advantage over dogs:
  - Dogs: 417 instances → 74.9% test mAP50
  - Humans: 3661 instances → expect 80%+ test mAP50

Usage:
  python3 person_optimization.py prepare
  python3 person_optimization.py train_baseline
  python3 person_optimization.py train_oversample
  python3 person_optimization.py train_yolov8s
  python3 person_optimization.py train_960px
  python3 person_optimization.py evaluate
  python3 person_optimization.py all
"""

import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter

# ═══════════════════════════ CONFIG ═══════════════════════════

BASE = Path("/home/praho/Documents/Job/BlajADER")
FUSED = BASE / "fused_dataset"
EXP = BASE / "person_experiment"
EXP.mkdir(exist_ok=True)

# Original class IDs in fused_dataset: 2=car, 4=dog, 7=person
PERSON_CLASS_ID = 7
TARGET_CLASS = 0  # Remap person to class 0 for binary detector

# Variants to test (informed by dog experiments)
VARIANTS = {
    "baseline": {
        "desc": "Blue channel, YOLOv8n, 640px - replicate dog success",
        "model": "yolov8n",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "augmentation": "default",
        "preprocessing": "blue",
        "oversample": 1,
        "image_filter": None,
    },
    "oversample": {
        "desc": "Blue channel, YOLOv8n, 640px, 2x oversampling (lighter than dogs)",
        "model": "yolov8n",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "augmentation": "default",
        "preprocessing": "blue",
        "oversample": 2,
        "image_filter": None,
    },
    "oversample_3x": {
        "desc": "Blue channel, YOLOv8n, 640px, 3x oversampling - EXACT DOG CONFIG",
        "model": "yolov8n",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "augmentation": "default",
        "preprocessing": "blue",
        "oversample": 3,
        "image_filter": None,
    },
    "red_channel": {
        "desc": "Red channel, YOLOv8n, 640px - test alternative thermal channel",
        "model": "yolov8n",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "augmentation": "default",
        "preprocessing": "red",
        "oversample": 1,
        "image_filter": None,
    },
    "green_channel": {
        "desc": "Green channel, YOLOv8n, 640px - test alternative thermal channel",
        "model": "yolov8n",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "augmentation": "default",
        "preprocessing": "green",
        "oversample": 1,
        "image_filter": None,
    },
    "rgb_full": {
        "desc": "Full RGB, YOLOv8n, 640px - no channel extraction",
        "model": "yolov8n",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "augmentation": "default",
        "preprocessing": None,
        "oversample": 1,
        "image_filter": None,
    },
    "thermal_only": {
        "desc": "Blue channel, YOLOv8n, 640px - THERMAL images only (640x512)",
        "model": "yolov8n",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "augmentation": "default",
        "preprocessing": "blue",
        "oversample": 1,
        "image_filter": "thermal",
    },
    "rgb_only": {
        "desc": "Full RGB, YOLOv8n, 640px - RGB images only (non-thermal)",
        "model": "yolov8n",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "augmentation": "default",
        "preprocessing": None,
        "oversample": 1,
        "image_filter": "rgb",
    },
    "yolov8s": {
        "desc": "Blue channel, YOLOv8s (11M params), 640px - large dataset justifies",
        "model": "yolov8s",
        "imgsz": 640,
        "epochs": 100,
        "batch": 12,
        "augmentation": "default",
        "preprocessing": "blue",
        "oversample": 1,
        "image_filter": None,
    },
    "960px": {
        "desc": "Blue channel, YOLOv8n, 960px - humans are larger targets",
        "model": "yolov8n",
        "imgsz": 960,
        "epochs": 100,
        "batch": 8,
        "augmentation": "default",
        "preprocessing": "blue",
        "oversample": 1,
        "image_filter": None,
    },
    "oversample_yolov8s": {
        "desc": "Blue channel, YOLOv8s, 640px, 2x oversampling - FINAL PUSH",
        "model": "yolov8s",
        "imgsz": 640,
        "epochs": 100,
        "batch": 12,
        "augmentation": "default",
        "preprocessing": "blue",
        "oversample": 2,
        "image_filter": None,
    },
    "rgb_full_3x": {
        "desc": "Full RGB (3 channels), YOLOv8n, 640px, 3x oversampling - SOLUTION ATTEMPT 1",
        "model": "yolov8n",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "augmentation": "default",
        "preprocessing": None,  # Use all RGB channels
        "oversample": 3,
        "image_filter": None,
    },
    "yolov8m_3x": {
        "desc": "Blue channel, YOLOv8m (25M params), 640px, 3x oversampling - SOLUTION ATTEMPT 2",
        "model": "yolov8m",
        "imgsz": 640,
        "epochs": 100,
        "batch": 10,  # Reduced batch for larger model
        "augmentation": "default",
        "preprocessing": "blue",
        "oversample": 3,
        "image_filter": None,
    },
}

RANDOM_SEED = 42


# ═══════════════════════════ FUNCTIONS ═══════════════════════════

def is_thermal_image(img_path):
    """Check if image is thermal based on dimensions (640x512 typical for thermal)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    # Thermal images are typically 640x512 or 512x640
    return (w == 640 and h == 512) or (w == 512 and h == 640)


def preprocess_image(img_path, preprocessing=None):
    """Apply preprocessing based on type: blue/red/green channel extraction or None for full RGB."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    if preprocessing == "blue":
        # Extract blue channel (BGR order: index 0)
        channel = img[:, :, 0]
        img_processed = cv2.merge([channel, channel, channel])
    elif preprocessing == "green":
        # Extract green channel (BGR order: index 1)
        channel = img[:, :, 1]
        img_processed = cv2.merge([channel, channel, channel])
    elif preprocessing == "red":
        # Extract red channel (BGR order: index 2)
        channel = img[:, :, 2]
        img_processed = cv2.merge([channel, channel, channel])
    elif preprocessing == "clahe":
        # CLAHE on blue channel (for backward compatibility)
        blue = img[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        blue = clahe.apply(blue)
        img_processed = cv2.merge([blue, blue, blue])
    elif preprocessing is None:
        # Full RGB, no preprocessing
        img_processed = img
    else:
        # Unknown preprocessing, return original
        img_processed = img
    
    return img_processed


def prepare_variant(variant_name, config):
    """Prepare dataset for a specific variant."""
    print(f"\n{'='*60}")
    print(f"Preparing: {variant_name}")
    print(f"Description: {config['desc']}")
    print(f"{'='*60}")
    
    variant_dir = EXP / variant_name
    variant_dir.mkdir(exist_ok=True)
    
    # Collect all thermal images with person annotations
    img_dir = FUSED / "images"
    label_dir = FUSED / "labels"
    
    person_images = []
    print("Scanning for ALL images with persons...")
    
    for img_path in tqdm(list(img_dir.glob("*.jpg"))):
        # Check if has person annotations
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        
        with open(label_path) as f:
            has_person = any(
                line.strip().split()[0] == str(PERSON_CLASS_ID)
                for line in f if line.strip()
            )
        
        if has_person:
            person_images.append({
                "img_path": img_path,
                "label_path": label_path,
            })
    
    print(f"Found {len(person_images)} images with persons (thermal + RGB)")
    
    # Apply image filter if specified
    image_filter = config.get("image_filter")
    if image_filter == "thermal":
        print("Filtering for THERMAL images only (640x512)...")
        person_images = [img for img in person_images if is_thermal_image(img["img_path"])]
        print(f"  → {len(person_images)} thermal images")
    elif image_filter == "rgb":
        print("Filtering for RGB images only (non-thermal)...")
        person_images = [img for img in person_images if not is_thermal_image(img["img_path"])]
        print(f"  → {len(person_images)} RGB images")
    
    # Create train/val/test split (75/15/10 - same as dogs)
    import random
    random.seed(RANDOM_SEED)
    random.shuffle(person_images)
    
    n = len(person_images)
    n_train = int(n * 0.75)
    n_val = int(n * 0.15)
    
    for i, item in enumerate(person_images):
        if i < n_train:
            item["split"] = "train"
        elif i < n_train + n_val:
            item["split"] = "val"
        else:
            item["split"] = "test"
    
    print(f"Split: train={n_train}, val={n_val}, test={n-n_train-n_val}")
    
    # Apply oversampling if needed (lighter than dogs)
    oversample_factor = config.get("oversample", 1)
    if oversample_factor > 1:
        print(f"Applying {oversample_factor}x oversampling to training set...")
        # Only oversample training images
        train_imgs = [img for img in person_images if img["split"] == "train"]
        other_imgs = [img for img in person_images if img["split"] != "train"]
        person_images = train_imgs * oversample_factor + other_imgs
        print(f"  → {len(person_images)} images after oversampling (train: {len(train_imgs) * oversample_factor})")
    
    # Process and save images
    splits_count = {"train": 0, "val": 0, "test": 0}
    preprocessing = config.get("preprocessing")
    
    for idx, item in enumerate(tqdm(person_images, desc="Processing")):
        split = item["split"]
        img_path = item["img_path"]
        label_path = item["label_path"]
        
        # Create output directories
        out_img_dir = variant_dir / "images" / split
        out_lbl_dir = variant_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply preprocessing (channel extraction or full RGB)
        processed_img = preprocess_image(img_path, preprocessing)
        if processed_img is None:
            continue
        
        # Generate unique filename
        base_name = img_path.stem
        out_img_path = out_img_dir / f"{base_name}_{idx}.jpg"
        out_lbl_path = out_lbl_dir / f"{base_name}_{idx}.txt"
        
        # Save image
        cv2.imwrite(str(out_img_path), processed_img)
        
        # Filter and remap labels (person only, remap to class 0)
        with open(label_path) as f:
            lines = f.readlines()
        
        person_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and int(parts[0]) == PERSON_CLASS_ID:
                # Remap to class 0
                person_lines.append(f"0 {' '.join(parts[1:])}\n")
        
        with open(out_lbl_path, "w") as f:
            f.writelines(person_lines)
        
        splits_count[split] += 1
    
    print(f"Dataset created: train={splits_count['train']}, "
          f"val={splits_count['val']}, test={splits_count['test']}")
    
    # Create dataset.yaml
    yaml_content = f"""# Person Detection - {variant_name}
# {config['desc']}

path: {variant_dir}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['person']
"""
    
    with open(variant_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"✓ Dataset YAML: {variant_dir / 'dataset.yaml'}")


def prepare_all():
    """Prepare datasets for all variants."""
    print("="*60)
    print("PERSON OPTIMIZATION - DATASET PREPARATION")
    print("="*60)
    print(f"Source dataset: 3661 person instances (9x more than dogs)")
    print(f"Expected performance: 80%+ test mAP50 (vs 74.9% for dogs)")
    print("="*60)
    
    for variant_name, config in VARIANTS.items():
        prepare_variant(variant_name, config)
    
    print("\n" + "="*60)
    print("ALL DATASETS PREPARED")
    print("="*60)


def train_variant(variant_name):
    """Train a specific variant."""
    from ultralytics import YOLO
    
    if variant_name not in VARIANTS:
        print(f"Unknown variant: {variant_name}")
        return
    
    config = VARIANTS[variant_name]
    variant_dir = EXP / variant_name
    dataset_yaml = variant_dir / "dataset.yaml"
    
    if not dataset_yaml.exists():
        print(f"Dataset not found for {variant_name}. Run 'prepare' first.")
        return
    
    # Check if already trained
    weights_dir = variant_dir / "runs" / "weights"
    last_checkpoint = weights_dir / "last.pt"
    best_checkpoint = weights_dir / "best.pt"
    
    # Check if training completed (best.pt exists with results.csv showing 100 epochs)
    results_csv = variant_dir / "runs" / "train" / "results.csv"
    if best_checkpoint.exists() and results_csv.exists():
        import pandas as pd
        try:
            df = pd.read_csv(results_csv)
            if len(df) >= 100:  # Completed full training
                print(f"{variant_name}: Already trained (100 epochs completed), skipping")
                return
        except:
            pass  # If CSV read fails, continue with training
    
    print(f"\n{'='*60}")
    print(f"Training: {variant_name}")
    print(f"Description: {config['desc']}")
    print(f"{'='*60}")
    
    model_name = config.get("model", "yolov8n")
    
    # Resume from checkpoint if exists, otherwise start fresh
    if last_checkpoint.exists():
        print(f"Resuming training from: {last_checkpoint}")
        model = YOLO(str(last_checkpoint))
    else:
        model = YOLO(f"{model_name}.pt")
    
    model.train(
        data=str(dataset_yaml),
        epochs=config.get("epochs", 100),
        imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 16),
        patience=20,
        project=str(variant_dir / "runs"),
        name="",
        pretrained=True,
        device=0,
        verbose=True,
        seed=RANDOM_SEED,
        workers=2,  # Reduce from default 8 to prevent process explosion
        # Augmentation settings (thermal-appropriate)
        degrees=0,  # No rotation for level thermal cameras
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )
    
    print(f"✓ {variant_name} training complete")


def evaluate_all():
    """Evaluate all trained variants and compare results."""
    from ultralytics import YOLO
    
    print("="*60)
    print("PERSON OPTIMIZATION - EVALUATION")
    print("="*60)
    
    results = {}
    
    for variant_name, config in VARIANTS.items():
        variant_dir = EXP / variant_name
        weights_path = variant_dir / "runs" / "weights" / "best.pt"
        dataset_yaml = variant_dir / "dataset.yaml"
        
        if not weights_path.exists():
            print(f"\n{variant_name}: No trained model found, skipping")
            continue
        
        print(f"\n{variant_name}: Evaluating on TEST set...")
        model = YOLO(str(weights_path))
        
        # Evaluate on test split
        r_test = model.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=config.get("imgsz", 640),
            batch=32,
            verbose=False
        )
        
        # Also get validation metrics
        r_val = model.val(
            data=str(dataset_yaml),
            split="val",
            imgsz=config.get("imgsz", 640),
            batch=32,
            verbose=False
        )
        
        results[variant_name] = {
            "desc": config["desc"],
            "val_mAP50": r_val.box.map50,
            "test_mAP50": r_test.box.map50,
            "test_precision": r_test.box.p[0] if len(r_test.box.p) > 0 else 0,
            "test_recall": r_test.box.r[0] if len(r_test.box.r) > 0 else 0,
            "gap": r_val.box.map50 - r_test.box.map50,  # Generalization gap
        }
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS COMPARISON (Test Set Performance)")
    print("="*60)
    print(f"{'Variant':<20} {'Test mAP50':>10} {'Val mAP50':>10} {'Gap':>8} {'P':>8} {'R':>8}")
    print("-"*70)
    
    baseline_test = results.get("baseline", {}).get("test_mAP50", 0)
    
    for variant_name, r in sorted(results.items(), key=lambda x: x[1]["test_mAP50"], reverse=True):
        diff = r["test_mAP50"] - baseline_test if baseline_test > 0 else 0
        diff_str = f"(+{diff*100:.1f}pp)" if diff > 0 else f"({diff*100:.1f}pp)" if diff < 0 else ""
        
        print(f"{variant_name:<20} "
              f"{r['test_mAP50']*100:>9.1f}% "
              f"{r['val_mAP50']*100:>9.1f}% "
              f"{r['gap']*100:>7.1f}pp "
              f"{r['test_precision']*100:>7.1f}% "
              f"{r['test_recall']*100:>7.1f}%")
        if diff_str:
            print(f"{'':20} {diff_str}")
    
    # Save results
    with open(EXP / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n✓ Results saved: {EXP / 'results.json'}")
    
    # Compare to dog performance
    print("\n" + "="*60)
    print("COMPARISON TO DOG DETECTOR")
    print("="*60)
    print(f"Dogs:    417 instances → 74.9% test mAP50")
    
    if results:
        best_variant = max(results.items(), key=lambda x: x[1]["test_mAP50"])
        best_test = best_variant[1]["test_mAP50"] * 100
        print(f"Persons: 3661 instances → {best_test:.1f}% test mAP50")
        print(f"\n🏆 BEST: {best_variant[0]}")
        print(f"   {best_variant[1]['desc']}")
        print(f"   Test: {best_test:.1f}% | Val: {best_variant[1]['val_mAP50']*100:.1f}% | Gap: {best_variant[1]['gap']*100:.1f}pp")
        
        if best_test >= 80.0:
            print(f"\n✅ TARGET ACHIEVED: {best_test:.1f}% ≥ 80%")
        else:
            print(f"\n⚠️  Target not reached: {best_test:.1f}% < 80% (gap: {80-best_test:.1f}pp)")
    
    print("="*60)


# ═══════════════════════════ MAIN ═══════════════════════════

if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if phase == "prepare":
        prepare_all()
    elif phase == "train_baseline":
        train_variant("baseline")
    elif phase == "train_oversample":
        train_variant("oversample")
    elif phase == "train_oversample_3x":
        train_variant("oversample_3x")
    elif phase == "train_red_channel":
        train_variant("red_channel")
    elif phase == "train_green_channel":
        train_variant("green_channel")
    elif phase == "train_rgb_full":
        train_variant("rgb_full")
    elif phase == "train_thermal_only":
        train_variant("thermal_only")
    elif phase == "train_rgb_only":
        train_variant("rgb_only")
    elif phase == "train_yolov8s":
        train_variant("yolov8s")
    elif phase == "train_960px":
        train_variant("960px")
    elif phase == "train_oversample_yolov8s":
        train_variant("oversample_yolov8s")
    elif phase == "train_rgb_full_3x":
        train_variant("rgb_full_3x")
    elif phase == "train_yolov8m_3x":
        train_variant("yolov8m_3x")
    elif phase == "train_channels":
        # Train all channel variants
        for v in ["baseline", "red_channel", "green_channel", "rgb_full"]:
            train_variant(v)
    elif phase == "train_image_types":
        # Train all image type variants
        for v in ["baseline", "thermal_only", "rgb_only"]:
            train_variant(v)
    elif phase == "train_all":
        for variant_name in VARIANTS.keys():
            train_variant(variant_name)
    elif phase == "evaluate":
        evaluate_all()
    elif phase == "all":
        prepare_all()
        for variant_name in VARIANTS.keys():
            train_variant(variant_name)
        evaluate_all()
    else:
        print(f"Unknown phase: {phase}")
        print("Usage: python3 person_optimization.py [prepare|train_<variant>|train_channels|train_image_types|train_all|evaluate|all]")
        print("Variants: baseline, oversample, oversample_3x, red_channel, green_channel, rgb_full, thermal_only, rgb_only, yolov8s, 960px, rgb_full_3x, yolov8m_3x")
