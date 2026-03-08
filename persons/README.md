# 🧍 Person Detector - Thermal Imaging

Person detection model for thermal (infrared) drone imagery using YOLOv8n.

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Validation mAP@0.5** | **54.7%** |
| **Validation mAP@0.5:0.95** | **28.3%** |
| **Precision** | 61.5% |
| **Recall** | 54.9% |
| **Parameters** | 3.0M |
| **Model Size** | 6.2 MB |

## 🎯 Model Details

- **Architecture**: YOLOv8n (nano)
- **Input**: RGB full-color thermal imagery
- **Image Size**: 640×640
- **Training Data**: 3,661 person instances
- **Data Augmentation**: 3x oversampling

## 🚀 Quick Start

### Installation

```bash
pip install ultralytics opencv-python
```

### Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run inference
results = model('your_thermal_image.jpg')

# Display results
results[0].show()
```

### Batch Processing

```python
import glob
from ultralytics import YOLO

model = YOLO('best.pt')

# Process all images in a folder
image_paths = glob.glob('images/*.jpg')
results = model(image_paths, stream=True)

for r in results:
    r.save(filename=f'output/{r.path}')
```

## 📋 Training Details

- **Epochs**: 100 (with early stopping)
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Learning Rate**: 0.002
- **Augmentations**: Mosaic, MixUp, HSV, Flip, Scale
- **Hardware**: NVIDIA RTX 4050 (6GB)
- **Training Time**: ~2.5 hours

## 🎓 Use Cases

- Search and rescue operations
- Crowd monitoring
- Security surveillance
- Wildlife/human conflict detection
- Border monitoring

## 📝 Citation

If you use this model in your research, please cite:

```bibtex
@software{person_detector_thermal_2026,
  title={Person Detection for Thermal UAV Imagery},
  author={Your Name},
  year={2026},
  url={https://github.com/PrahoLama/person-detector-thermal}
}
```

## 📄 License

MIT License - feel free to use for research and commercial applications.

## 🔗 Related Projects

- [Dog Detector (Thermal)](https://github.com/PrahoLama/dog-detector-thermal) - 74.9% mAP@0.5
- [Car Detector (Thermal)](https://github.com/PrahoLama/car-detector-thermal) - 78.8% mAP@0.5

## ⚠️ Limitations

- Performance varies with thermal image quality
- Challenging in crowded scenes with occlusion
- Best results with standing/walking persons
- Lower accuracy than car/dog detection due to:
  - Greater pose variation
  - Smaller thermal signature
  - More occlusion cases

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

**Model Version**: v1.0  
**Last Updated**: March 2026
