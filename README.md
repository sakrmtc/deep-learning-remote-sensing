# 🛰️ Deep Learning for Remote Sensing Image Analysis

> **Master's Research Project** — Deep learning-based classification and image-to-image translation applied to remote sensing / terrain imagery.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [1. Terrain Classifier (CNN)](#1-terrain-classifier-cnn)
  - [2. Residual U-Net (Res-U-Net)](#2-residual-u-net-res-u-net)
- [Dataset](#dataset)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [References](#references)
- [Author](#author)

---

## 📖 Overview

This repository contains the deep learning models developed during my Master's program for analyzing and processing terrain images (mountains, sand, rivers, and sea/coastal areas).

The work covers two major tasks:
1. **Multi-class terrain classification** using a Convolutional Neural Network (CNN).
2. **Image-to-image translation** (image restoration/denoising) using a **Residual U-Net** architecture.

Both models were trained on Google Colab using Google Drive for dataset management.

---

## 🚀 Projects

---

### 1. Terrain Classifier (CNN)

**File:** `classifier_sea.ipynb`

#### 🎯 Objective
Classify terrain images into **3 categories**: Mountain, Sand, and River.

#### 🏗️ Architecture

```
Input (128×128 RGB)
  → Conv2D(64, 5×5, stride=2) + LeakyReLU(0.2) + Dropout(0.25)
  → Conv2D(128, 5×5, stride=2) + LeakyReLU(0.2) + Dropout(0.25)
  → Flatten
  → Dense(3, softmax)
```

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D | (64, 64, 64) | 4,864 |
| Conv2D | (32, 32, 128) | 204,928 |
| Flatten | (131,072) | 0 |
| Dense | (3) | 393,219 |
| **Total** | | **603,011** |

#### 📊 Training Results

| Split | Samples | Accuracy |
|-------|---------|----------|
| Train | 1,901 | 100% |
| Validation | 476 | 100% |
| Test | 265 | **100%** |

- **Test Loss:** `0.00015`
- **Test Accuracy:** `1.00` (100%)
- **Epochs:** 20
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Cross-Entropy

#### 🏷️ Classes
| Label | Class |
|-------|-------|
| 0 | Mountain |
| 1 | Sand |
| 2 | River |

---

### 2. Residual U-Net (Res-U-Net)

**File:** `res_u_net_paper.ipynb`

#### 🎯 Objective
Perform **image-to-image translation** (restoration/denoising) on terrain images — mapping noisy/flipped input images to clean output images.

#### 🏗️ Architecture

A **U-Net** architecture enhanced with **Residual Blocks** in the encoder path:

```
Input (128×128×3)
  ↓ Encoder (Residual Blocks + MaxPooling)
    ResBlock(32) → MaxPool
    ResBlock(64) → MaxPool
    ResBlock(128) → MaxPool
    ResBlock(256) → MaxPool
    ResBlock(512) → MaxPool
  ↓ Bottleneck
    Conv2D(1024) × 2
  ↑ Decoder (Transposed Convolutions + Skip Connections)
    Conv2DTranspose(512) + Concat → Conv2D × 2
    Conv2DTranspose(256) + Concat → Conv2D × 2
    Conv2DTranspose(128) + Concat → Conv2D × 2
    Conv2DTranspose(64)  + Concat → Conv2D × 2
    Conv2DTranspose(32)  + Concat → Conv2D × 2
  ↓ Output
    Conv2DTranspose(3) + Sigmoid
```

| Model Stats | Value |
|------------|-------|
| Total Parameters | 34,766,307 (~132.62 MB) |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Epochs | 200 |

#### 📊 Training Results (selected epochs)

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.0116 | 0.0087 |
| 10 | 0.0066 | 0.0065 |
| 50 | 0.0036 | 0.0050 |
| 100 | 0.0030 | 0.0049 |
| 200 | 0.0026 | 0.0049 |

#### 📐 Dataset Split

| Split | Samples |
|-------|---------|
| Train | 20,000 |
| Validation | 4,000 |
| Test | 1,000 |
| **Total** | **25,000** |

---

## 📁 Dataset

Datasets were stored on **Google Drive** and accessed via Google Colab:

- **Classifier Dataset:** Images organized into `mountain/`, `sand/`, `river/` folders (`.jpg` format, resized to 128×128)
- **Res-U-Net Dataset:** Paired input/output images stored in `flipped/input/` and `flipped/output/` folders (25,000 pairs, 128×128 RGB)

> ⚠️ **Note:** Datasets are not included in this repository due to size constraints. Please contact the author for access.

---

## 📄 References

- JARS-240130G_online.pdf — Reference paper for the Res-U-Net architecture.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.

---

## ⚙️ Requirements

```
Python >= 3.8
TensorFlow >= 2.x
Keras
NumPy
OpenCV (cv2)
scikit-learn
matplotlib
scikit-image
tensorflow-datasets
```

Install all dependencies:
```bash
pip install tensorflow numpy opencv-python scikit-learn matplotlib scikit-image tensorflow-datasets
```

---

## ▶️ How to Run

1. Open the notebooks in **Google Colab** (recommended for GPU access)
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/gdrive/')
   ```
3. Update dataset paths in the notebook to match your Drive folder structure
4. Run all cells sequentially

---

## 👤 Author

**Sakrm** — Master's Student  
*Deep Learning · Remote Sensing · Computer Vision*

📧 Contact: [your email here]  
🔗 GitHub: [your GitHub profile here]

---

## 📜 License

This project is for **academic purposes only**.  
© 2026 — All rights reserved.
