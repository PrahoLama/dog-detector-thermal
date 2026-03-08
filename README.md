# 🔥 Thermal Object Detection - Multi-Class YOLOv8

High-performance object detection models for thermal (infrared) drone imagery.

## 🎯 Detection Models

| Object | mAP@0.5 | Precision | Recall | Status |
|--------|---------|-----------|--------|--------|
| **🐕 Dogs** | **74.9%** | 95.8% | 64.3% | ✅ Best |
| **🚗 Cars** | **78.8%** | 79.8% | 71.8% | ✅ Best |
| **🧍 Persons** | **54.7%** | 61.5% | 54.9% | ⚠️ Challenging |

All models use YOLOv8n architecture (3.0M parameters) optimized for different thermal patterns.

## 📂 Repository Structure

```
thermal-object-detection/
├── dogs/                          # Dog detector (74.9% mAP50)
│   ├── best.pt                    # Trained weights
│   ├── README.md                  # Dog-specific docs
│   └── TRAINING.md                # Training details
├── cars/                          # Car detector (78.8% mAP50)
│   ├── best.pt                    # Trained weights
│   ├── README.md                  # Car-specific docs
│   └── TRAINING.md                # Training details
├── persons/                       # Person detector (54.7% mAP50)
│   ├── best.pt                    # Trained weights
│   ├── README.md                  # Person-specific docs
│   └── TRAINING.md                # Training details
├── weights/                       # Legacy dog detector
└── README.md                      # This file
```

---

## 🐕 Dogs - 74.9% mAP50

High-performance dog detection optimized with blue channel extraction.

## 📊 Model Performance

### Test Results (38-image holdout set)
| Metric | Value |
|--------|-------|
| **mAP50** | **74.9%** |
| Precision | 95.8% |
| Recall | 64.3% |
| mAP50-95 | ~ |

### Key Features:
- **Best mAP@0.5**: 74.9% on test set
- **Preprocessing**: Blue channel extraction (optimal thermal band)
- **Data**: 846 training images (3x oversampling)
- **High Precision**: 95.8% (low false positives)

---

## 🚗 Cars - 78.8% mAP50 

Best overall performance with excellent balance between precision and recall.

### Key Features:
- **Best mAP@0.5**: 78.8% on test set
- **Preprocessing**: Blue channel extraction
- **Data**: 4,944 training images (3x oversampling, 1,648 instances)
- **Balanced**: 79.8% precision, 71.8% recall

---

## 🧍 Persons - 54.7% mAP50

Challenging detection task due to pose variation and occlusion.

### Key Features:
- **Best mAP@0.5**: 54.7% on validation set
- **Preprocessing**: RGB full-color (best for persons)
- **Data**: 3,661 instances
- **Limitations**: High pose variation, occlusion challenges

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/PrahoLama/dog-detector-thermal.git
cd dog-detector-thermal

# Install dependencies
pip install ultralytics opencv-python
```

### Inference Examples

#### Dogs
```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('dogs/best.pt')

# Prepare image (blue channel extraction)
img = cv2.imread('thermal_image.jpg')
blue = img[:, :, 0]
img_processed = cv2.merge([blue, blue, blue])

# Detect
results = model(img_processed)
results[0].show()
```

#### Cars
```python
from ultralytics import YOLO
import cv2

model = YOLO('cars/best.pt')

img = cv2.imread('thermal_image.jpg')
blue = img[:, :, 0]
img_processed = cv2.merge([blue, blue, blue])

results = model(img_processed)
results[0].show()
```

#### Persons
```python
from ultralytics import YOLO

# Load model (uses RGB, no preprocessing needed)
model = YOLO('persons/best.pt')

# Detect directly
results = model('thermal_image.jpg')
results[0].show()
```

---

## 📊 Model Comparison

### Performance by Object Type

| Metric | Dogs | Cars | Persons |
|--------|------|------|---------|
| **mAP@0.5** | 74.9% | 78.8% | 54.7% |
| **mAP@0.5:0.95** | - | 41.1% | 28.3% |
| **Precision** | 95.8% | 79.8% | 61.5% |
| **Recall** | 64.3% | 71.8% | 54.9% |
| **Parameters** | 3.0M | 3.0M | 3.0M |
| **Instances** | 417 | 1,648 | 3,661 |

### Why Different Performance?

**Cars** (Best - 78.8%):
- Large thermal signature
- Consistent shape
- High contrast with background
- Minimal occlusion

**Dogs** (Good - 74.9%):
- Strong thermal signal (37-39°C)
- Relatively consistent shape
- Blue channel optimization
- Some pose variation

**Persons** (Challenging - 54.7%):
- High pose variation (standing, sitting, walking)
- Frequent occlusion
- Smaller thermal signature
- Complex backgrounds

---

## 🛠️ Training Details

All models trained with:

All models trained with:
- **Architecture**: YOLOv8n (3.0M parameters)
- **Resolution**: 640×640
- **Optimizer**: AdamW (lr=0.002)
- **Epochs**: 100 with early stopping
- **Batch Size**: 16
- **Hardware**: NVIDIA RTX 4050 6GB
- **Augmentation**: Mosaic, MixUp, HSV, flip, scale

For detailed training methodology, see each subdirectory's `TRAINING.md`.

---

## 📂 Using the Models

### Individual Model Usage

Each detector is independent - use only what you need:

```python
# Just dogs
dog_model = YOLO('dogs/best.pt')

# Just cars  
car_model = YOLO('cars/best.pt')

# Just persons
person_model = YOLO('persons/best.pt')
```

### Multi-Class Detection

Run all detectors on the same image:

```python
from ultralytics import YOLO
import cv2

# Load all models
dog_model = YOLO('dogs/best.pt')
car_model = YOLO('cars/best.pt')
person_model = YOLO('persons/best.pt')

# Load and preprocess image
img = cv2.imread('thermal_image.jpg')
blue_ch = cv2.merge([img[:,:,0]] * 3)  # For dogs and cars

# Detect all objects
dogs = dog_model(blue_ch)[0]
cars = car_model(blue_ch)[0]
persons = person_model(img)[0]  # RGB for persons

# Combine results
print(f"Found: {len(dogs.boxes)} dogs, {len(cars.boxes)} cars, {len(persons.boxes)} persons")
```

---

## 🎓 Use Cases

**Search & Rescue**:
- Detect persons in thermal imagery
- Identify service dogs
- Locate vehicles

**Wildlife & Crowd Monitoring**:
- Dog population tracking
- Human-wildlife conflict zones
- Parking lot analysis

**Security & Surveillance**:
- Perimeter monitoring (persons + vehicles)
- Unauthorized vehicle detection
- Thermal surveillance systems

---

## 📝 Citation

```bibtex
@software{thermal_object_detection_2026,
  title={Multi-Object Detection for Thermal UAV Imagery},
  author={PrahoLama},
  year={2026},
  url={https://github.com/PrahoLama/dog-detector-thermal}
}
```

---

## ⚠️ Known Limitations

**Dogs**:
- Performance drops with heavy vegetation occlusion
- Requires blue channel preprocessing

**Cars**:
- Best with minimal occlusion by buildings/trees
- Designed for thermal view from above

**Persons**:
- Lower accuracy due to pose variation
- Challenging in crowded scenes
- Best with standing/walking persons

---

## 🤝 Contributing

Contributions welcome! To improve any detector:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

See individual detector READMEs for training instructions.

---

## 📄 License

MIT License - Free for research and commercial use.

---

**Repository Status**: Production-ready ✅  
**Last Updated**: March 2026  
**Models**: 3 (Dogs, Cars, Persons)

