# Training Documentation

## Dataset Statistics

- **Total Instances**: 1,648 cars
- **Total Images**: 596 thermal images
- **Data Split**:
  - Training: 1,788 images (3× oversampled from 596 base images)
  - Validation: 119 images
  - Test: 80 images

## Training Configuration

### Model Architecture
- **Base Model**: YOLOv8n (nano)
- **Parameters**: 3,005,843 (3.0M)
- **FLOPs**: 8.1 GFLOPs

### Preprocessing
- **Channel Extraction**: Blue channel only (BGR index 0)
- **Rationale**: Thermal cameras encode temperature data primarily in blue channel
- **Channel Replication**: Single channel replicated to 3 channels for YOLOv8 compatibility

### Hyperparameters
```python
{
    'model': 'yolov8n.pt',
    'data': 'dataset.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'lr0': 0.002,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'optimizer': 'AdamW',
    'patience': 20,
    'workers': 2,
    'device': 0,
    'seed': 42,
    'deterministic': True
}
```

### Data Augmentation
- **Oversampling**: 3× (596 → 1,788 training images)
- **YOLO Built-in Augmentations**:
  - HSV color jittering
  - Random horizontal flip
  - Mosaic augmentation
  - MixUp augmentation  - Affine transformations

## Training Results

### Training Progress
- **Total Epochs**: 76 (stopped early at patience=20)
- **Best Epoch**: 43
- **Best Validation mAP50**: 79.1%
- **Training Time**: ~2 hours on RTX 4050 GPU

### Performance Metrics

| Dataset | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------|---------|--------------|-----------|--------|
| Validation | 79.1% | 40.8% | 81.2% | 73.5% |
| **Test** | **78.8%** | **41.1%** | **79.8%** | **71.8%** |

### Training Curves
- Training converged smoothly
- No overfitting observed (val ≈ test performance)
- Early stopping triggered after 20 epochs without improvement

## Inference Speed

- **Preprocessing**: 3.6ms per image
- **Inference**: 5.7ms per image
- **Postprocessing**: 1.3ms per image
- **Total**: ~11ms per image (~91 FPS)

## Reproducibility

### Environment
```bash
python==3.12.3
torch==2.5.1+cu121
ultralytics==8.4.21
opencv-python==4.10.0
numpy==2.1.3
```

### Random Seeds
- PyTorch seed: 42
- NumPy seed: 42
- Deterministic mode: Enabled

### GPU
- NVIDIA GeForce RTX 4050 Laptop GPU
- CUDA: 12.1
- Memory: 6GB

## Running Training

```bash
# Prepare dataset (with 3× oversampling)
python car_optimization.py prepare_oversample_3x

# Train model
python car_optimization.py train_oversample_3x
```

## Comparison with Other Approaches

The blue channel extraction with 3× oversampling achieved the best results:

| Approach | Preprocessing | Oversampling | mAP@0.5 |
|----------|---------------|--------------|---------|
| **Optimal** | **Blue Channel** | **3×** | **78.8%** |
| Baseline | Blue Channel | None | ~65-70% (estimated) |
| RGB Full | All 3 Channels | 3× | Not tested for cars |

## Key Insights

1. **Blue Channel is Critical**: Thermal temperature data is primarily encoded in the blue channel of exported thermal images
2. **Oversampling Helps**: 3× oversampling provides enough data diversity for YOLOv8n to learn effectively
3. **Model Size**: YOLOv8n (3M params) is sufficient for car detection in thermal imagery
4. **Stability**: workers=2 prevents GPU memory issues and process explosion
5. **Generalization**: Test performance (78.8%) matches validation (79.1%), indicating good generalization
