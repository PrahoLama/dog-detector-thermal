#!/usr/bin/env python3
"""
Evaluation script for dog detector model.

Usage:
    python scripts/evaluate.py --model weights/best.pt --data config/dataset.yaml --split test
"""

import argparse
from ultralytics import YOLO


def evaluate_model(model_path, data_yaml, split='test'):
    """
    Evaluate model on test or validation set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML configuration
        split: Dataset split to evaluate on ('test' or 'val')
    """
    print("="*70)
    print(f"EVALUATING MODEL ON {split.upper()} SET")
    print("="*70)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run validation
    print(f"Evaluating on {split} split...")
    results = model.val(
        data=data_yaml,
        split=split,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"mAP50:      {results.box.map50*100:.1f}%")
    print(f"mAP50-95:   {results.box.map*100:.1f}%")
    print(f"Precision:  {results.box.mp*100:.1f}%")
    print(f"Recall:     {results.box.mr*100:.1f}%")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate dog detector model'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='weights/best.pt',
        help='Path to model weights (default: weights/best.pt)'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to dataset YAML configuration'
    )
    parser.add_argument(
        '--split', 
        type=str, 
        default='test',
        choices=['test', 'val'],
        help='Dataset split to evaluate on (default: test)'
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split
    )


if __name__ == '__main__':
    main()
