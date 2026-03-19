# AE-UVNet: An Attention Enhanced hybrid architecture using U-Net and V-Net for Multimodal Brain tumour Segmentation

## Overview

AE-UVNet is a hybrid dual-encoder architecture for volumetric medical image segmentation.

![without_notion.drawio](C:\Users\ROG\Downloads\without_notion.drawio.png)

It integrates:

- Residual U-Net encoder (local detail)
- V-Net encoder (context aggregation)
- Transformer bottleneck (global attention)
- SE attention (channel recalibration)

The network is designed for multimodal MRI segmentation tasks such as BraTS.

## Project Scheme

## Training Environment

All experiments were conducted using **Google Colab** with GPU acceleration.

### Hardware

- GPU: NVIDIA A100 (Colab High-RAM)
- RAM: ~40GB
- Storage: Google Drive (optional for checkpoints)

---

### Software Environment

*   **tensorflow**: 2.21.0
*   **keras**: 3.3.3
*   **segmentation-models-3D**: 1.1.1
*   **plotly**: 5.24.1
*   **split-folders**: 0.6.1

### Notes

- The project is tested on **Google Colab GPU runtime**
- Using other environments may require version adjustments
- NumPy ≥ 2.0 may cause compatibility issues with some Keras versions

## Usage 

```python
from model import build_model

model = build_model(
    model_name='uvnet',
    input_shape=(128,128,128,3),
    n_classes=4
)
```

### Available models

- unet_plain

- vnet_plain

- unet_transformer

- vnet_transformer
- uvnet

## Dataset

This project uses the **BraTS 2020 dataset** for training and evaluation.

The dataset includes multi-modal 3D MRI scans with expert annotations for brain tumour segmentation.

### Modalities used
- T1ce (contrast-enhanced T1)
- T2
- FLAIR

### Task
Voxel-wise segmentation of tumour subregions:

- WT (Whole Tumour)
- TC (Tumour Core)
-  ET (Enhancing Tumour)

### Dataset Access

The BraTS dataset is publicly available but requires registration.

You can download it from the official kaggle website:

https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

### Data Disclaimer

This repository does not include any medical data.

Users must download the dataset from the official source and comply with its license and usage policies.