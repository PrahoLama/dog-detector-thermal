# 🏋️ Training Details - Person Detector

Complete training methodology and optimization process.

## 📊 Dataset Statistics

**Source**: Thermal UAV imagery  
**Total Instances**: 3,661 persons  
**Split Ratio**: 80% train, 10% validation, 10% test

### Distribution:
- Training: ~2,929 instances
- Validation: ~366 instances
- Test: ~366 instances

## 🔧 Preprocessing

### 1. Image Format
- **Input**: Thermal infrared images (pseudo-colored)
- **Channel**: RGB full-color (all channels retained)
- **Resolution**: Original resolution maintained, resized to 640×640 for training

### 2. Data Augmentation
Applied 3x oversampling with:
- **Mosaic**: 4-image composition (probability: 1.0)
- **MixUp**: Alpha blending (probability: 0.1)
- **HSV Augmentation**: 
  - Hue: ±0.015
  - Saturation: ±0.7
  - Value: ±0.4
- **Flip**: Horizontal (probability: 0.5)
- **Scale**: ±50% random scaling
- **Translation**: ±10% random translation
- **Rotation**: Not applied (preserves orientation)

## 🎯 Model Architecture

**Base Model**: YOLOv8n (nano variant)

**Architecture Details**:
- **Backbone**: CSPDarknet with C2f modules
- **Neck**: PAN (Path Aggregation Network)
- **Head**: Decoupled detection head
- **Parameters**: 3,012,537 (3.0M)
- **FLOPs**: 8.1G

## 🏃 Training Configuration

### Hyperparameters:
```python
{
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'optimizer': 'AdamW',
    'lr0': 0.002,          # Initial learning rate
    'lrf': 0.01,           # Final learning rate factor
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,            # Box loss gain
    'cls': 0.5,            # Class loss gain
    'dfl': 1.5,            # DFL loss gain
    'patience': 20,        # Early stopping patience
    'workers': 2,          # DataLoader workers (GPU memory limited)
    'device': 0,           # CUDA device
}
```

### Training Schedule:
- **Learning Rate**: Cosine decay from 0.002 to 0.00002
- **Warmup**: 3 epochs linear warmup
- **Early Stopping**: 20 epochs patience on validation mAP

## 📈 Training Results

### Final Metrics (Epoch 82 - Early Stopped):
| Metric | Value |
|--------|-------|
| Train Box Loss | 1.229 |
| Train Class Loss | 0.720 |
| Train DFL Loss | 0.968 |
| Val mAP@0.5 | **54.7%** |
| Val mAP@0.5:0.95 | 28.3% |
| Precision | 61.5% |
| Recall | 54.9% |

### Training Evolution:
- **Best Epoch**: 82 (out of 100)
- **Training Time**: ~2.5 hours
- **GPU Memory**: ~4.2GB peak usage
- **Convergence**: Smooth convergence, early stopped at epoch 82

## 🔬 Optimization Attempts

### Approaches Tested:
1. **Thermal Blue Channel** → 49.4% mAP50 (baseline)
2. **RGB Full-Color** → **54.7% mAP50** (best) ✅
3. **YOLOv8s (larger)** → 54.7% mAP50 (no improvement)
4. **YOLOv8x (largest)** → 51.0% mAP50 (worse, overfitting)

### Key Findings:
- RGB full-color performs better than single-channel extraction for persons
- Model capacity (nano vs small vs xlarge) has minimal impact
- **Dataset difficulty** is the limiting factor, not model size
- 3x oversampling provides significant boost

## 🎓 Comparison with Other Objects

| Object | mAP@0.5 | Difficulty |
|--------|---------|-----------|
| **Dogs** | 74.9% | Easy |
| **Cars** | 78.8% | Easy |
| **Persons** | 54.7% | Hard |

### Why Persons are Harder:
1. **Pose Variation**: Standing, sitting, walking, crouching
2. **Occlusion**: Often partially hidden
3. **Scale Variation**: Adults, children, different distances
4. **Thermal Signature**: Smaller and less distinct than cars
5. **Background Clutter**: More varied environments

## 🛠️ Hardware Requirements

**Minimum**:
- GPU: 6GB VRAM (NVIDIA RTX 4050 tested)
- RAM: 16GB system memory
- Storage: 10GB for dataset + model

**Recommended**:
- GPU: 8GB+ VRAM
- RAM: 32GB system memory
- Storage: SSD for faster data loading

## 🚀 Inference Speed

| Hardware | FPS (640×640) |
|----------|---------------|
| RTX 4050 | ~85 FPS |
| CPU (i7) | ~8 FPS |

## 📝 Training Command

```bash
python person_optimization.py
```

Or directly with YOLO:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='person_experiment/rgb_full_3x/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.002,
    patience=20,
    workers=2,
    device=0
)
```

## 🔍 Future Improvements

1. **More Training Data**: Current 3,661 instances may be insufficient
2. **Better Annotations**: Review and refine person bounding boxes
3. **Specialized Architecture**: Person-specific model design
4. **Pose-Aware Detection**: Multi-pose training strategy
5. **Temporal Information**: Use video sequences for tracking
6. **Multi-Scale Testing**: Test at multiple resolutions

## 📚 References

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Original YOLO: https://pjreddie.com/darknet/yolo/
- Thermal Image Enhancement: Various preprocessing techniques tested

---

**Training Date**: March 2026  
**Framework**: Ultralytics YOLOv8  
**GPU**: NVIDIA RTX 4050 6GB
