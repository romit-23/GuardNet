# GuardNet: Violence Detection in Public Surveillance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![IEEE Published](https://img.shields.io/badge/IEEE-Published-green.svg)](https://ieeexplore.ieee.org/)

> **An Efficient Deep Learning Approach for Real-Time Public Safety via Violence Recognition**

GuardNet is a state-of-the-art deep learning framework for automated violence detection in CCTV surveillance systems. By combining ConvNeXt-Tiny CNN architecture with LSTM networks and temporal attention mechanisms, GuardNet achieves **97.6% accuracy** on real-world surveillance datasets.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Model Performance](#model-performance)
- [Citation](#citation)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## 🎯 Overview

GuardNet addresses the growing need for automated violence detection in smart cities and public safety applications. Traditional manual CCTV monitoring is labor-intensive and prone to human error. Our hybrid deep learning approach provides:

- **Real-time violence detection** in video streams
- **High accuracy** (97.6%) with minimal false positives
- **Scalable architecture** suitable for deployment in urban surveillance systems
- **Robust performance** across diverse real-world conditions

### Published Research

This project has been published in **IEEE** as:

> *GuardNet: An Efficient Deep Learning Approach for Real-Time Public Safety via Violence Recognition*

Authors: Raj Veer, Romit Addagatla, Harsh Mogal, Suyash Ayachit, Arundhati Das  
Institution: SIES GST, Mumbai, India

## ✨ Key Features

- **Hybrid Architecture**: Combines ConvNeXt-Tiny + LSTM + Multi-Head Attention
- **Temporal Modeling**: Captures both spatial features and temporal dependencies
- **Memory-Efficient**: Chunked processing for GPU memory optimization
- **Mixed Precision Training**: Faster training with lower memory footprint
- **Progressive Unfreezing**: Transfer learning with gradual fine-tuning
- **Data Augmentation**: Robust to environmental variations and lighting conditions
- **Multi-Dataset Training**: Trained on combined real-world surveillance datasets

## 🏗️ Architecture

GuardNet employs a two-stage deep learning pipeline:

### 1. Feature Extraction (ConvNeXt-Tiny)
- Pre-trained on ImageNet-1K
- Extracts 768-dimensional features per frame
- Chunked processing (4 frames at a time) for memory efficiency
- Mixed precision computation with automatic casting

### 2. Temporal Sequence Modeling
- **Bidirectional LSTM**: Captures long-term temporal dependencies (hidden_dim=384)
- **Multi-Head Self-Attention**: Transformer-style attention (4 heads) for temporal context
- **Global Temporal Pooling**: Weighted aggregation of frame representations
- **Classification Head**: Fully connected layers with GELU activation and LayerNorm

```
Input Video (16 frames)
    ↓
ConvNeXt-Tiny Feature Extractor
    ↓ (768-dim features per frame)
Bidirectional LSTM
    ↓
Multi-Head Attention (4 heads)
    ↓
Global Temporal Pooling
    ↓
FC Layers + Classification
    ↓
Output: Violence/Non-Violence
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/guardnet.git
cd guardnet
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy pandas scikit-learn matplotlib seaborn
```

### Required Libraries

```python
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 📊 Dataset

GuardNet is trained and validated on three diverse datasets:

### 1. Real-Life Violence Situations Dataset
- **Size**: 2,000 videos (1,000 violent + 1,000 non-violent)
- **Source**: YouTube real-world street fights and normal activities
- **Characteristics**: Diverse environments, varying lighting, multiple camera angles

### 2. Smart City CCTV Violence Detection Dataset (SCVD)
- **Type**: Real CCTV surveillance footage
- **Classes**: Violence, Normal, Weapons detection
- **Unique Feature**: First video-based weapons detection dataset
- **Focus**: Urban surveillance scenarios with handheld objects as weapons

### 3. Combined Dataset
- **Classes**: Fight, NonFight, HockeyFight, MovieFight
- **Purpose**: Multi-class violence detection
- **Split**: Training and validation sets

### Dataset Structure

```
datasets/
├── real_life_violence/
│   ├── Violence/
│   └── NonViolence/
├── scvd/
│   ├── Train/
│   │   ├── Violence/
│   │   └── Normal/
│   └── Test/
│       ├── Violence/
│       └── Normal/
└── combined/
    ├── train/
    │   ├── Fight/
    │   └── NonFight/
    └── val/
        ├── Fight/
        └── NonFight/
```

### Data Preprocessing

- **Frame Sampling**: Uniform sampling of 16 frames per video
- **Resolution**: 224×224 pixels
- **Normalization**: Min-max scaling (0-1 range)
- **Augmentation**: 
  - Random horizontal flip (p=0.5)
  - Random crop-resize (80-100% scale)
  - Random rotation (±15°)
  - Color jitter (brightness/contrast)

## 💻 Usage

### Training

```python
# Set your dataset paths
VIOLENT_PATHS = [
    "path/to/violence/videos1",
    "path/to/violence/videos2"
]

NON_VIOLENT_PATHS = [
    "path/to/normal/videos1",
    "path/to/normal/videos2"
]

# Run training
python final-model.py
```

**Training Parameters:**
- Batch Size: 8
- Epochs: 100 (with early stopping)
- Learning Rate: 3e-4 (AdamW optimizer)
- Patience: 10 epochs
- Input: 16 frames @ 224×224 resolution

### Inference

```python
import torch
from inference import preprocess_video, ViolenceDetectionModel

# Load model
model = ViolenceDetectionModel()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Process video
video_path = "path/to/test/video.mp4"
frames = preprocess_video(video_path)

# Predict
with torch.no_grad():
    output = model(frames)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1)
    
    if prediction.item() == 1:
        print(f"Violence detected! Confidence: {probabilities[0][1].item():.2%}")
    else:
        print(f"No violence. Confidence: {probabilities[0][0].item():.2%}")
```

### Real-Time CCTV Monitoring

```python
import cv2

cap = cv2.VideoCapture(0)  # or RTSP stream URL
buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    buffer.append(frame)
    
    if len(buffer) == 16:
        # Process buffer with model
        prediction = detect_violence(buffer)
        if prediction == "Violence":
            # Trigger alert
            send_alert()
        
        buffer = buffer[8:]  # Sliding window
    
    cv2.imshow('CCTV', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📈 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **97.60%** |
| **Precision (Violence)** | 96.26% |
| **Recall (Violence)** | 99.04% |
| **F1-Score (Violence)** | 97.63% |
| **Precision (Non-Violence)** | 99.01% |
| **Recall (Non-Violence)** | 96.17% |
| **F1-Score (Non-Violence)** | 97.57% |
| **AUC-ROC** | **1.00** |

### Confusion Matrix

|  | Predicted Non-Violence | Predicted Violence |
|---|---|---|
| **Actual Non-Violence** | 201 (96.17%) | 8 (3.83%) |
| **Actual Violence** | 2 (0.96%) | 206 (99.04%) |

**Total Test Samples**: 417 videos

### Training Curves

- Training converged at epoch 24
- Final training loss: 0.079
- Final validation loss: 0.082
- No overfitting observed (validation F1: 0.976)

## 🔬 Model Performance

### Comparison with Other Architectures

| Model | Real-Life Violence | Smart CCTV | Combined Dataset |
|-------|-------------------|------------|-----------------|
| 3D CNN + GRU + Attention | 86.98% | 58.65% | 76.93% |
| I3D + LSTM + Attention | 94.01% | 66.35% | 88.24% |
| 3D CNN + LSTM + Attention | 82.81% | 46.63% | 70.34% |
| EfficientNet + LSTM + Attention | 97.90% | **99.30%** | 95.35% |
| **ConvNeXt + LSTM + Attention** | **97.90%** | 97.60% | **95.47%** |

### Key Advantages

✅ **Superior temporal modeling** with bidirectional LSTM  
✅ **Attention mechanism** captures critical motion patterns  
✅ **Lightweight architecture** suitable for edge deployment  
✅ **Robust to occlusions** and varying lighting conditions  
✅ **Real-time capable** with GPU acceleration  

## 📝 Citation

If you use GuardNet in your research, please cite:

```bibtex
@inproceedings{veer2024guardnet,
  title={GuardNet: An Efficient Deep Learning Approach for Real-Time Public Safety via Violence Recognition},
  author={Veer, Raj and Addagatla, Romit and Mogal, Harsh and Ayachit, Suyash and Das, Arundhati},
  booktitle={IEEE Conference Proceedings},
  year={2024},
  organization={IEEE},
  address={Mumbai, India}
}
```

## 👥 Authors

- **Raj Veer** - [rajuttamveer2003@gmail.com](mailto:rajuttamveer2003@gmail.com)
- **Romit Addagatla** - [addromit2307@gmail.com](mailto:addromit2307@gmail.com)
- **Harsh Mogal** - [harshmogal@gmail.com](mailto:harshmogal@gmail.com)
- **Suyash Ayachit** - [suyash.ayachit29@gmail.com](mailto:suyash.ayachit29@gmail.com)
- **Arundhati Das** (Advisor) - [arundhatid@sies.edu.in](mailto:arundhatid@sies.edu.in)

**Institution**: Department of Artificial Intelligence and Machine Learning  
SIES Graduate School of Technology, Mumbai, India

## 🙏 Acknowledgments

- **Dataset Providers**: Real-Life Violence Situations, SCVD, Combined Dataset
- **Pre-trained Models**: ConvNeXt-Tiny (ImageNet-1K)
- **Frameworks**: PyTorch, OpenCV, scikit-learn
- **Computing Resources**: CUDA GPU infrastructure
- **Research Guidance**: Dr. Arundhati Das, SIES GST

## 🔗 Links

- **IEEE Paper**: [[Link to published paper](https://ieeexplore.ieee.org/document/11284226)]

## 🌟 Future Work

- [ ] Real-time video stream processing optimization
- [ ] Mobile deployment (TensorFlow Lite/ONNX)
- [ ] Multi-person violence detection
- [ ] Weapon detection integration
- [ ] Edge device deployment (NVIDIA Jetson)
- [ ] Cloud-based distributed inference
- [ ] Anomaly detection for other surveillance tasks

## 📧 Contact

For questions, collaborations, or commercial inquiries:

- **Primary Contact**: Romit A - addromit2307@gmail.com
- **Academic Queries**: Dr. Arundhati Das - arundhatid@sies.edu.in

---

**⚠️ Disclaimer**: This system is designed for public safety applications. Users must ensure compliance with local laws and regulations regarding surveillance and privacy. The authors are not responsible for misuse of this technology.

**🌐 Contribute**: We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Made with ❤️ by the GuardNet Team at SIES GST, Mumbai
