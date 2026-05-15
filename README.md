# VisDrone Object Detection Pipeline

A comprehensive computer vision pipeline for detecting humans and vehicles in aerial drone imagery using YOLOv8.

## 📋 Overview

This project implements an end-to-end object detection system trained on the **VisDrone Dataset**, specifically optimized for detecting pedestrians and vehicles from drone-based imagery. It leverages the YOLOv8 nano model for fast and efficient inference while maintaining competitive accuracy.

### Key Features
- 🚁 **Drone Footage Optimization**: Trained specifically on aerial/drone imagery
- 👥 **Multi-class Detection**: Detects humans and vehicles with confidence scores
- ⚡ **Lightweight Model**: Uses YOLOv8 Nano for quick inference
- 📊 **Performance Metrics**: Includes mAP, precision, and recall evaluation
- 🎨 **Visual Outputs**: Color-coded detection visualization with bounding boxes
- 📈 **Human Counting**: Automatic pedestrian counting in images

---

## 🛠️ Prerequisites

- **Python** 3.8+
- **CUDA** (optional, for GPU acceleration)
- **Google Colab** (recommended) or local machine with sufficient disk space
- Kaggle API credentials for dataset download

### System Requirements
- **GPU Memory**: 4GB+ (for training)
- **Storage**: ~50GB (for VisDrone dataset)
- **Processor**: Multi-core recommended

---

## 📦 Installation

### Step 1: Install Dependencies

The notebook automatically installs required packages:
```bash
pip install ultralytics
pip install kaggle
```

**Library Versions**:
- `ultralytics` - YOLOv8 framework
- `kaggle` - Dataset download utility
- `opencv-python` - Image processing
- `matplotlib` - Visualization
- `pyyaml` - Configuration parsing

### Step 2: Configure Kaggle API

Set up your Kaggle credentials for dataset access:
```bash
mkdir -p ~/.kaggle
echo YOUR_KAGGLE_API_KEY > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

---

## 📊 Dataset

### VisDrone Dataset
The project uses the **VisDrone 2019 Object Detection Dataset**, which includes:
- **Training Set**: High-resolution drone images with object annotations
- **Validation Set**: For hyperparameter tuning
- **Test Set**: For final model evaluation

#### Dataset Statistics
- **Source**: banuprasadb/visdrone-dataset (Kaggle)
- **Format**: YOLO format annotations
- **Classes**:
  - 0: Pedestrian (Human)
  - 1: People
  - 2: Bicycle
  - 3: Car
  - 4: Van
  - 5: Truck
  - 6: Tricycle
  - 7: Awning-tricycle
  - 8: Bus
  - 9: Motor

#### Download & Extract
```bash
kaggle datasets download -d banuprasadb/visdrone-dataset
unzip visdrone-dataset.zip -d /content/visdrone
```

---

## 📁 Project Structure

```
Computer_Vision_Pipeline.ipynb
├── 1. Setup & Dependencies
│   ├── Package installation (ultralytics, kaggle)
│   └── Kaggle API configuration
│
├── 2. Dataset Preparation
│   ├── Download VisDrone dataset
│   ├── Extract and organize files
│   └── Verify dataset structure
│
├── 3. Data Exploration
│   ├── Visualize sample images
│   ├── Draw bounding boxes
│   └── Display annotations
│
├── 4. Configuration
│   ├── Update YAML paths
│   └── Prepare dataset config
│
├── 5. Model Training
│   ├── Initialize YOLOv8 Nano
│   ├── Train on VisDrone data
│   └── Save best weights
│
├── 6. Inference & Detection
│   ├── Load trained model
│   ├── Run predictions
│   ├── Visualize results
│   └── Count detections
│
└── 7. Evaluation
    ├── Calculate mAP metrics
    ├── Compute precision/recall
    └── Performance analysis
```

---

## 🚀 Usage Guide

### 1. **Setup Environment**
Execute the first cell to install all dependencies:
```python
!pip install ultralytics
!pip install kaggle
```

### 2. **Download Dataset**
Configure Kaggle and download the dataset:
```python
!kaggle datasets download -d banuprasadb/visdrone-dataset
!unzip visdrone-dataset.zip -d /content/visdrone
```

### 3. **Explore Dataset**
Visualize sample annotations with bounding boxes:
```python
img_path = '/content/visdrone/VisDrone_Dataset/VisDrone2019-DET-train/images/sample.jpg'
ann_path = '/content/visdrone/VisDrone_Dataset/VisDrone2019-DET-train/labels/sample.txt'
visualize_sample(img_path, ann_path)
```

### 4. **Configure Dataset**
Update the YAML configuration with correct paths:
```python
data_yaml['path'] = '/content/visdrone/VisDrone_Dataset'
```

### 5. **Train Model**
Start training with YOLOv8 Nano:
```python
model = YOLO('yolov8n.pt')
model.train(
    data=yaml_path,
    epochs=30,
    imgsz=640,
    batch=4,
    project='/content/drive/MyDrive/yolo_runs',
    name='visdrone_exp'
)
```

### 6. **Run Inference**
Detect humans and vehicles in images:
```python
detect_humans_and_cars(image_path, show=True)
```

### 7. **Evaluate Performance**
Calculate model metrics:
```python
metrics = model.val(data='visdrone.yaml')
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

---

## 🔍 Core Components

### 1. **Data Visualization Function** (`visualize_sample`)
Displays images with bounding boxes drawn from YOLO format annotations.

**Input Format** (YOLO):
```
class_id x_center_norm y_center_norm width_norm height_norm
```

**Functionality**:
- Load images using OpenCV
- Parse annotation files
- Convert normalized coordinates to pixel coordinates
- Draw rectangles and class labels
- Display with Matplotlib

### 2. **Model Training**
- **Architecture**: YOLOv8 Nano (lightweight variant)
- **Input Size**: 640×640 pixels
- **Batch Size**: 4 (adjustable for available GPU memory)
- **Epochs**: 30 (configurable)
- **Optimization**: SGD with momentum

### 3. **Inference Pipeline** (`detect_humans_and_cars`)
Performs real-time object detection with filtering:

**Features**:
- Loads trained model weights
- Runs inference on input images
- Filters detections by class (humans and cars only)
- Counts total pedestrians detected
- Visualizes results with bounding boxes and confidence scores
- Displays human count overlay

**Output**:
- Annotated image with bounding boxes
- Confidence scores for each detection
- Total human count

### 4. **Model Evaluation**
Validates trained model on validation dataset:
- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: Mean Average Precision across IoU thresholds (0.5:0.95)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

---

## 📈 Training Details

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Model Type | YOLOv8 Nano |
| Input Resolution | 640×640 |
| Batch Size | 4 |
| Epochs | 30 |
| Learning Rate | Auto (YOLOv8 default) |
| Augmentation | Standard (YOLOv8 defaults) |

### Output Location
Training results are saved to:
```
/content/drive/MyDrive/yolo_runs/visdrone_exp/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
└── confusion_matrix.png
```

---

## 🎯 Detection Classes

The model is specifically configured to detect:

| Class ID | Class Name | Color |
|----------|-----------|-------|
| 0 | Human/Pedestrian | Green (0, 255, 0) |
| 3 | Car | Blue (255, 0, 0) |

Other classes (bicycles, vans, trucks, etc.) are filtered out during inference.

---

## 📊 Performance Metrics

After training and validation, the model provides:

```python
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")      # Standard metric
print(f"mAP50-95: {metrics.box.map:.3f}")     # COCO metric
print(f"Precision: {metrics.box.p.mean():.3f}")
print(f"Recall: {metrics.box.r.mean():.3f}")
```

### Expected Performance (YOLOv8 Nano on VisDrone)
- **mAP50**: ~35-45% (depends on training configuration)
- **Inference Speed**: <50ms per image (GPU)
- **Model Size**: ~7MB

---

## 💡 Tips & Best Practices

### For Better Accuracy
1. **Increase Training Epochs**: Change from 30 to 50-100
2. **Use Larger Model**: Replace `yolov8n.pt` with `yolov8s.pt` or `yolov8m.pt`
3. **Adjust Batch Size**: Increase batch size if GPU memory allows
4. **Data Augmentation**: YOLOv8 includes automatic augmentation

### For Faster Inference
1. Use YOLOv8 Nano (`yolov8n.pt`) - already configured
2. Lower input resolution from 640 to 416 or 320
3. Deploy on GPU for best performance
4. Use INT8 quantization for edge devices

### Memory Optimization
- **Google Colab**: Usually sufficient for training with batch_size=4
- **Local GPU**: Adjust batch size based on available VRAM
- **CPU-Only**: Training will be slow; use smaller models or datasets

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError` for dataset paths
- **Solution**: Verify paths match your extraction directory

**Issue**: CUDA out of memory during training
- **Solution**: Reduce batch size or use a smaller model variant

**Issue**: Low detection accuracy
- **Solution**: Increase training epochs, use data augmentation, verify annotations

**Issue**: Kaggle API authentication fails
- **Solution**: Ensure API key is correctly placed in `~/.kaggle/access_token`

---

## 📚 References

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **VisDrone Dataset**: http://aiskyeye.com/
- **Kaggle Dataset**: https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset

---

## 🤝 Contributing

Improvements and contributions are welcome! Consider:
- Testing different YOLOv8 variants (small, medium, large)
- Adding additional object classes
- Implementing post-processing filters
- Optimizing for specific use cases

---

## 📝 License

This project uses:
- **YOLOv8**: Licensed under AGPL-3.0
- **VisDrone Dataset**: Available for research purposes

Please refer to original licenses before using in commercial applications.

---

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review YOLOv8 documentation
3. Consult VisDrone dataset documentation

---

**Last Updated**: May 2026  
**Notebook Version**: Computer_Vision_Pipeline.ipynb
