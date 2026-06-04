# 🌍 Multispectral Cloud & Cloud Shadow Segmentation using Deep Learning

## 📌 Project Overview

This project implements a lightweight deep learning pipeline for automated cloud, cloud shadow, and clear sky segmentation from Sentinel-2 multispectral satellite imagery.

The primary objective of this work is to explore AI-based alternatives to traditional sensor-dependent QA masking techniques commonly used in satellite image preprocessing pipelines.

The model performs semantic segmentation on multispectral satellite data and is optimized for efficient execution on consumer-grade GPUs.

---

# 🚀 Features

- 🌥️ Cloud Detection
- 🌑 Cloud Shadow Detection
- ☀️ Clear Sky Segmentation
- 🛰️ Sentinel-2 Multispectral Support
- ⚡ Lightweight Deep Learning Architecture
- 🎯 Pixel-wise Semantic Segmentation
- 🧠 GPU Accelerated Training (CUDA)
- 📊 Visualization & Overlay Rendering
- 🧩 Modular Training & Inference Pipeline

---

# 🧠 Architecture

This project uses a hybrid segmentation architecture:

MobileNetV2 Encoder + U-Net Style Decoder

### Encoder
- MobileNetV2 is used as the backbone feature extractor
- Lightweight and computationally efficient
- Extracts deep semantic features from multispectral imagery

### Decoder
- U-Net style decoder upsamples feature maps
- Produces pixel-wise segmentation outputs
- Generates semantic masks for:
  - Cloud
  - Cloud Shadow
  - Clear Sky

---

# 🛰️ Dataset

The project was trained using publicly available Sentinel-2 cloud segmentation datasets.

## Dataset Sources

### CloudSEN12
A global dataset for cloud and cloud shadow semantic understanding in Sentinel-2 imagery.

### GDSD (Globally Distributed Sentinel-2 Cloud and Cloud Shadow Dataset)
Provides annotated Sentinel-2 imagery for cloud and cloud shadow segmentation.

## Dataset Links

### CloudSEN12
:contentReference[oaicite:0]{index=0}

### GDSD Dataset
:contentReference[oaicite:1]{index=1}

---

# 🏷️ Dataset Labels

| Label Value | Class |
|---|---|
| 0 | Fill / Invalid |
| 64 | Cloud Shadow |
| 128 | Clear Sky |
| 255 | Cloud |

---

# 📦 Why the Dataset is NOT Included

The dataset used in this project consists of large Sentinel-2 TIFF imagery files which collectively occupy several gigabytes of storage.

Due to:
- GitHub file size limitations
- Repository performance considerations
- Storage efficiency
- Public availability of the datasets

the raw satellite imagery and masks are intentionally excluded from this repository.

The datasets can be downloaded directly from their official providers using the links above.

---

# 🏋️ Training Details

## Hardware Used

- GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU
- VRAM: 8 GB
- CUDA Accelerated Training

## Training Optimizations

- Mixed Precision Training (AMP)
- cuDNN Benchmark Optimization
- Early Stopping
- Multi-worker Data Loading
- GPU Pinned Memory

## Training Summary

- Trained on approximately 3500+ image-mask pairs
- Early stopping triggered after 28 epochs
- Successfully learned:
  - Cloud regions
  - Cloud shadows
  - Clear sky regions

---

# 📂 Project Structure

```text
cloud_mask_project/

├── configs/
├── datasets/
├── inference/
├── models/
├── training/
├── utils/
├── assets/

├── best_cloud_model.pth
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Technologies Used

- Python
- PyTorch
- TorchVision
- CUDA
- NumPy
- tifffile
- OpenCV
- Matplotlib

---

# ▶️ Training

Run training using:

```bash
python main.py --stage train
```

---

# 🔍 Inference

Run prediction using:

```bash
python inference/predict.py
```

---

# 📊 Visualization

The pipeline supports:
- RGB satellite visualization
- Semantic segmentation masks
- Cloud overlays
- Cloud shadow visualization

---

# 🎯 Applications

- Satellite QA Masking
- Remote Sensing
- Earth Observation
- Agricultural Monitoring
- Disaster Monitoring
- Weather Analytics
- Geospatial AI

---

# 📈 Future Improvements

- Full U-Net Skip Connections
- Attention U-Net
- Transformer-based Segmentation
- Multi-class Dice Loss
- Real-time Edge Deployment
- Temporal Cloud Tracking

---

# 📚 Acknowledgements

This project utilizes publicly available Sentinel-2 datasets and builds upon research contributions from the remote sensing and geospatial AI community.

Special thanks to:
- CloudSEN12 dataset contributors
- GDSD dataset contributors
- Sentinel-2 / Copernicus Program
- PyTorch open-source community

---

# 📌 Notes

This project is intended for research, educational, and experimental purposes related to satellite image processing and AI-based cloud masking.
