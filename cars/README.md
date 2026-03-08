# 🚗 Car Detector - Thermal Imaging

High-performance car detection model for thermal (infrared) drone imagery using YOLOv8n.

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Test mAP@0.5** | **78.8%** |
| **Test mAP@0.5:0.95** | **41.1%** |
| **Precision** | 79.8% |
| **Recall** | 71.8% |
| **Parameters** | 3.0M |
| **Model Size** | 6.2 MB |

## 🎯 Model Details

- **Architecture**: YOLOv8n (nano)
- **Input Channel**: Blue channel extraction (optimal thermal band)
- **Image Size**: 640×640
- **Training Dataset**: 1,648 car instances from 596 thermal images
- **Data Augmentation**: 3× oversampling
- **Training Split**: 75% train / 15% val / 10% test
- **Training**: 100 epochs (early stopped at epoch 43)

## 🚀 Quick Start

### Installation

\`\`\`bash
pip install ultralytics opencv-python numpy
\`\`\`

### Inference

\`\`\`python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('best.pt')

# Run inference on thermal image
results = model('thermal_image.jpg')

# Display results
results[0].show()
\`\`\`

### Preprocessing (Blue Channel Extraction)

The model expects images preprocessed with blue channel extraction:

\`\`\`python
import cv2

def preprocess_thermal(img_path):
    """Extract blue channel from thermal image."""
    img = cv2.imread(img_path)
    blue_channel = img[:, :, 0]  # BGR format
    # Replicate to 3 channels for YOLO
    return cv2.merge([blue_channel, blue_channel, blue_channel])

# Preprocess and detect
processed = preprocess_thermal('thermal_image.jpg')
results = model(processed)
\`\`\`

## 📁 Repository Contents

- `best.pt` - Trained YOLOv8n model weights (6.2 MB)
- `car_optimization.py` - Complete training script
- `TRAINING.md` - Detailed training information
- `examples/` - Example inference code

## 🔬 Training Approach

This model uses the **blue channel extraction** method optimized specifically for thermal imaging:

1. **Blue Channel Selection**: Thermal cameras often encode temperature data in the blue channel of BGR format
2. **3× Oversampling**: Training images augmented through intelligent oversampling
3. **YOLOv8n Architecture**: Lightweight yet accurate (3M parameters)
4. **Optimized Hyperparameters**:
   - Learning rate: 0.002
   - Batch size: 16
   - Patience: 20 epochs
   - Workers: 2 (GPU stability)

## 📈 Training Curves

Best validation mAP@0.5 achieved at **epoch 43 (79.1%)**, final test: **78.8%**.

## 🌡️ Thermal Imaging Notes

This model is specifically designed for **thermal (infrared) imagery** from UAV/drone cameras. It:
- Works best with FLIR and similar thermal cameras
- Optimized for aerial/elevated viewpoints
- Handles various temperature ranges
- Robust to lighting conditions (day/night)

## 📝 Citation

If you use this model in your research, please cite:

\`\`\`bibtex
@software{thermal_car_detector_2026,
  author = {PrahoLama},
  title = {Car Detector for Thermal UAV Imagery},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/PrahoLama/car-detector-thermal}
}
\`\`\`

## 🤝 Related Projects

- [Dog Detector - Thermal](https://github.com/PrahoLama/dog-detector-thermal) - 74.9% mAP@0.5

## 📄 License

MIT License

## 🔗 Links

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Training Documentation](TRAINING.md)
