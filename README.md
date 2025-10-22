# 🔮 SmartPrice AI: Vision-Language Commerce Platform

<div align="center">

![SmartPrice AI](https://img.shields.io/badge/SmartPrice_AI-2025-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![DeBERTa](https://img.shields.io/badge/DeBERTa_v3-base-yellow?style=for-the-badge)
![EfficientNet](https://img.shields.io/badge/EfficientNet_v2-S-blue?style=for-the-badge)
![GPU](https://img.shields.io/badge/CUDA-Compatible-76B900?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-Proprietary-purple?style=for-the-badge)

**State-of-the-art deep learning model for accurate product price prediction using multi-modal analysis of text descriptions and images**

[Features](#features) • [Architecture](#model-architecture) • [Installation](#installation) • [Usage](#usage) • [Training Details](#training-details)

</div>

---

## 🎯 Overview

SmartPrice AI harnesses the power of advanced deep learning models to provide accurate price predictions for e-commerce products. By combining DeBERTa-v3's superior text understanding with EfficientNet-v2's efficient image processing, it delivers reliable pricing insights for diverse product categories.

### ✨ Key Highlights

- 🤖 **Multi-Modal Analysis** - Combined text and image processing
- 🎯 **High Accuracy** - State-of-the-art prediction performance
- ⚡ **Fast Inference** - GPU-accelerated with CPU fallback
- 📊 **Batch Processing** - Efficient bulk prediction support
- 🔄 **Production Ready** - Optimized deployment pipeline
- 🎨 **Feature Fusion** - Advanced cross-attention mechanism
- 💻 **Hardware Flexible** - Works on both GPU and CPU
- 📈 **Scalable** - Handles diverse product categories

---

## Performance Examples

Sample predictions from our model:
```
Product: Reese's Chocolate Peanut Butter Shell Topping Bottle, 7.25 oz
Predicted Price: $17.79

Product: McCormick Culinary Vanilla Extract, 32 fl oz
Predicted Price: $20.68

Product: Vlasic Snack'mm's Kosher Dill 16 Oz (Pack of 2)
Predicted Price: $21.08
```

Average prediction time:
- GPU: ~0.1 seconds per product
- CPU: ~0.5 seconds per product

## Features

- Multi-modal price prediction (text + image)
- State-of-the-art transformer architecture
- GPU-accelerated inference
- Production-ready prediction pipeline
- Batch processing support

## Model Architecture

- **Text Processing**: DeBERTa-v3-base for advanced text understanding
- **Image Processing**: EfficientNet-v2-S for efficient image feature extraction
- **Feature Fusion**: Dual cross-attention mechanism
- **Resolution**: 224x224 image input
- **Text Length**: Up to 128 tokens


- **Image Processing**: EfficientNet-v2-S for efficient image feature extraction
- **Feature Fusion**: Dual cross-attention mechanism
- **Resolution**: 224x224 image input
- **Text Length**: Up to 128 tokens

## Project Structure

```
SmartPrice-AI/
├── config/
│   └── quick_clean_config.yaml     # Model configuration
├── models/
│   └── best_model.pth             # Trained weights
├── src/
│   ├── advanced_efficient_predictor.py  # Model architecture
│   ├── dataset.py                 # Data processing
│   └── predict_pipeline.py        # Deployment pipeline
├── data/
│   └── README.md                  # Data directory info
└── requirements.txt               # Dependencies
```

## Requirements

<div align="center">
  <img src="https://www.python.org/static/community_logos/python-logo-generic.svg" width="300px">
</div>

### Core Requirements
- Python 3.8+
- CUDA compatible GPU (optional, falls back to CPU)

### Key Libraries
| Category | Libraries |
|----------|-----------|
| **Deep Learning** | ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch) ![transformers](https://img.shields.io/badge/🤗_transformers-latest-blue?style=flat-square) |
| **Computer Vision** | ![Pillow](https://img.shields.io/badge/Pillow-10.0%2B-2B6DAA?style=flat-square) ![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat-square) |
| **Data Processing** | ![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy) ![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas) |
| **NLP** | ![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green?style=flat-square) |
| **Utilities** | ![tqdm](https://img.shields.io/badge/tqdm-latest-FFC107?style=flat-square) ![PyYAML](https://img.shields.io/badge/PyYAML-6.0%2B-gray?style=flat-square) |

### Detailed Dependencies
```
├── torch>=2.0.0
├── transformers>=4.30.0
├── pillow>=10.0.0
├── opencv-python>=4.8.0
├── numpy>=1.24.0
├── pandas>=2.0.0
├── nltk>=3.8.0
├── pyyaml>=6.0.0
├── tqdm>=4.65.0
└── requests>=2.31.0
```

## Installation

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/VinayDattarao/SmartPrice-AI.git
cd SmartPrice-AI
```

2. Download required files:
   - Visit [Releases](https://github.com/VinayDattarao/SmartPrice-AI/releases)
   - Download these files:
     - `best_model.zip` - Contains the trained model
     - `tokenizer.zip` - Contains tokenizer files
     - `dataset.7z.001`, `dataset.7z.002`, `dataset.7z.003` (optional, for training)

3. Extract the files:
```bash
# Create directories if they don't exist
mkdir -p models/tokenizer

# Extract model and tokenizer (Windows PowerShell)
Expand-Archive -Path best_model.zip -DestinationPath models/
Expand-Archive -Path tokenizer.zip -DestinationPath models/tokenizer/

# For dataset (using 7-Zip, optional)
# Place all dataset.7z.* files in the same folder and extract the first file
# 7-Zip will automatically combine all parts
```

4. Set up Python environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

5. Verify Installation:
```bash
python test_installation.py
```

The test script checks:
- Python version compatibility
- CUDA availability
- Required files presence
- Model and tokenizer loading
- Configuration validation

### Common Issues and Solutions

1. "Model file not found":
   - Make sure you extracted `best_model.zip` to the `models/` folder
   - Verify `models/best_model.pth` exists

2. "Tokenizer files missing":
   - Check that `tokenizer.zip` was extracted to `models/tokenizer/`
   - Required files: `tokenizer.json`, `vocab.json`, `special_tokens_map.json`

3. CUDA/GPU issues:
   - Install NVIDIA drivers if using GPU
   - Model works on CPU if no GPU available
   - Check GPU status: `python -c "import torch; print(torch.cuda.is_available())"` 

## Usage

### Quick Start

```python
from src.predict_pipeline import PricePredictionPipeline

# Initialize the pipeline
pipeline = PricePredictionPipeline()

# Make a single prediction
text = "Organic Green Tea, Premium Quality, 100 Tea Bags"
image_path = "path/to/product/image.jpg"
price = pipeline.predict(text, image_path)
print(f"Predicted Price: ${price:.2f}")

# Batch prediction
texts = ["Product 1 description", "Product 2 description"]
images = ["path/to/image1.jpg", "path/to/image2.jpg"]
prices = pipeline.predict_batch(texts, images)
```

### Model Performance

- Optimized for e-commerce product pricing
- Handles diverse product categories
- Fast inference time

## Training Details

### Dataset
- Training samples: 75,000 products
- Validation samples: 25,000 products
- Test samples: 75,000 products
- Data includes: Product descriptions, images, and prices

### Training Infrastructure
- GPU: NVIDIA CUDA-compatible GPU (8GB+ VRAM recommended)
- Training time: ~8-10 hours on a modern GPU
- Batch size: 64 samples per batch
- Mixed precision training (FP16) for efficiency

### Model Architecture
1. Text Processing:
   - Model: DeBERTa-v3-base
   - Max length: 128 tokens
   - Hidden size: 512
   - Attention heads: 8

2. Image Processing:
   - Model: EfficientNet-v2-S
   - Input size: 224x224 pixels
   - Normalized RGB images
   - Advanced augmentation pipeline

3. Feature Fusion:
   - Dual cross-attention mechanism
   - Bidirectional text-image attention
   - Residual connections
   - Layer normalization

### Training Process
1. Initial phase:
   - Learning rate: 0.0003
   - Weight decay: 0.002
   - OneCycleLR scheduler
   - Gradient clipping at 1.0

2. Optimization:
   - Mixed precision training
   - Label smoothing (0.1)
   - Custom loss function (SMAPE + MSE)
   - Early stopping with patience=5

Training configuration details can be found in `config/quick_clean_config.yaml`.

## License

Copyright (c) 2025 Vinay Datta Rao

All rights reserved. This project and its contents are protected under copyright law.
No part of this project may be reproduced, distributed, or transmitted in any form or by any means without the prior written permission of the copyright holder.

## Acknowledgments

- Image processing based on EfficientNet
- Text processing based on DeBERTa-v3
- Training infrastructure using PyTorch
