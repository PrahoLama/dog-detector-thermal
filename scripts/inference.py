#!/usr/bin/env python3
"""
Inference script for dog detection on thermal imagery using blue channel extraction.

Usage:
    python scripts/inference.py --image path/to/thermal_image.jpg --model weights/best.pt
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO


def extract_blue_channel(img_path):
    """
    Extract blue channel from thermal image and replicate to 3 channels.
    
    Args:
        img_path: Path to input image
        
    Returns:
        Processed image with blue channel replicated to RGB
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Extract blue channel (OpenCV uses BGR order)
    blue = img[:, :, 0]
    
    # Replicate to 3 channels for YOLO compatibility
    img_processed = cv2.merge([blue, blue, blue])
    
    return img_processed


def run_inference(model_path, image_path, conf_threshold=0.25, save_results=True):
    """
    Run inference on a single image or directory of images.
    
    Args:
        model_path: Path to trained YOLO model weights
        image_path: Path to image or directory of images
        conf_threshold: Confidence threshold for detections
        save_results: Whether to save annotated results
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Process single image or directory
    image_path = Path(image_path)
    
    if image_path.is_file():
        images = [image_path]
    elif image_path.is_dir():
        images = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
    else:
        raise ValueError(f"Invalid image path: {image_path}")
    
    print(f"Processing {len(images)} image(s)...")
    
    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        
        # Apply blue channel extraction
        img_processed = extract_blue_channel(img_path)
        
        # Run inference
        results = model(img_processed, conf=conf_threshold, verbose=False)
        
        # Display results
        result = results[0]
        num_detections = len(result.boxes)
        print(f"  Detections: {num_detections}")
        
        if num_detections > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                print(f"    - Confidence: {conf:.2%}")
        
        # Save annotated image
        if save_results:
            output_dir = Path('results')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"detected_{img_path.name}"
            
            # Get annotated image
            annotated = result.plot()
            cv2.imwrite(str(output_path), annotated)
            print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Dog detection inference on thermal imagery'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='weights/best.pt',
        help='Path to model weights (default: weights/best.pt)'
    )
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Path to image or directory of images'
    )
    parser.add_argument(
        '--conf', 
        type=float, 
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='Do not save annotated results'
    )
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model,
        image_path=args.image,
        conf_threshold=args.conf,
        save_results=not args.no_save
    )
    
    print("\n✓ Inference complete!")


if __name__ == '__main__':
    main()
