# Efficient-FIQA

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=sunwei925/Efficient-FIQA)
[![GitHub stars](https://img.shields.io/github/stars/sunwei925/Efficient-FIQA)](https://github.com/sunwei925/Efficient-FIQA)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-brightgreen?logo=PyTorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/UIQA)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sunwei925/Efficient-FIQA)
[![arXiv](https://img.shields.io/badge/arXiv-2507.15709-red?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2507.15709)

**🏆 🥇 Winner Solution for [ICCV VQualA 2025 Face Image Quality Assessment Challenge](https://codalab.lisn.upsaclay.fr/competitions/23017)**

*Official Implementation of "Efficient Face Image Quality Assessment via Self-training and Knowledge Distillation"*

[📖 Paper](https://arxiv.org/abs/2507.15709) | [🤗 Demo](https://huggingface.co/spaces/sunwei925/Efficient-FIQA) | [📊 Challenge Results](https://codalab.lisn.upsaclay.fr/competitions/23017)

</div>

---

## 📋 Table of Contents

- [🎯 Introduction](#-introduction)
- [🏆 Challenge Results](#-challenge-results)
- [📦 Installation](#-installation)
- [📊 Dataset](#-dataset)
- [🔧 Training](#-training)
- [🧪 Testing](#-testing)
- [📚 Citation](#-citation)

---

## 🎯 Introduction

**Face Image Quality Assessment (FIQA)** is crucial for various face-related applications such as face recognition, face detection, and biometric systems. While significant progress has been made in FIQA research, the computational complexity remains a key bottleneck for real-world deployment.

This repository presents **Efficient-FIQA**, a novel approach that achieves state-of-the-art performance with extremely low computational overhead through:

- **🔬 Self-training Strategy**: Enhances teacher model capacity using pseudo-labeled data
- **🎓 Knowledge Distillation**: Transfers knowledge from powerful teacher to lightweight student
- **⚡ Efficient Architecture**: Student model achieves comparable performance with minimal computational cost

### 🏆 Key Achievements

- **🥇 1st Place** in ICCV VQualA 2025 FIQA Challenge

---

## 🏆 Challenge Results

| Rank | Team | Score | GFLOPs | Params (M) |
|:----:|------|:-----:|:------:|:----------:|
| 🥇 **1** | **ECNU-SJTU VQA Team (Ours)** | **0.9664** | **0.3313** | **1.1796** |
| 2 | MediaForensics | 0.9624 | 0.4687 | 1.5189 |
| 3 | Next | 0.9583 | 0.4533 | 1.2224 |
| 4 | ATHENAFace | 0.9566 | 0.4985 | 2.0916 |
| 5 | NJUPT-IQA-Group | 0.9547 | 0.4860 | 3.7171 |
| 6 | ECNU VIS Lab | 0.9406 | 0.4923 | 3.2805 |

*Score = (SRCC + PLCC) / 2*

For more results on the ICCV VQualA 2025 FIQA Challenge, please refer to the challenge report.

---


## 📦 Installation

### Requirements

- Python >= 3.9
- PyTorch >= 1.13
- CUDA >= 11.0 (for GPU training)

### Environment Setup

```bash
# Create and activate conda environment
conda create -n EfficientFIQA python=3.9
conda activate EfficientFIQA

# Install other dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset

### Download Links

| File | Google Drive | Baidu Yun |
|------|-------------|-----------|
| Training Dataset | [Download](https://drive.google.com/file/d/1FpylY9uVOfdKw5vI6UduMviUMIiDfK-7/view) | [Download](https://pan.baidu.com/s/18nk2BzrykyHusfTDX5w7xg?pwd=edts) |
| Ground Truth Scores | [Download](https://drive.google.com/file/d/1UQ8m4gIPg5X2LC3ugifWGhul0LP86_9b/view) | - |
| Validation Dataset | [Download](https://drive.google.com/file/d/1UM8IgjFjf6O3hIwhVqfFaLkHMjtMOvK2/view) | [Download](https://pan.baidu.com/s/1UtTXwgb13B7lxDFjtLPx5g) |

### Dataset Structure

```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── train.csv  # Format: image_name,score
```

---

## 🔧 Training

### Step 1: Train Teacher Model

1. **Configure paths** in `config_SwinB.py`:
```python
# Data paths
data_dir: str = '/path/to/your/training/images'
csv_path: str = 'data_file/train.csv'

# Model save path
model_save_dir: str = '/path/to/save/teacher/model'
```

2. **Start training**:
```bash
python train_teacher_model.py
```

### Step 2: Train Student Model

The teacher model is first used to generate pseudo-labels for unlabeled images to enhance training data. Since we cannot provide the original unlabeled images due to copyright restrictions, we use [GFIQA-20K](https://database.mmsp-kn.de/gfiqa-20k-database.html) as a representative example dataset.

1. **Generate pseudo-labels** using teacher model:
```python
# Configure in test_unlabeled_images.py
image_dir = '/path/to/GFIQA/images'
output_csv = 'data_file/gfiqa_results.csv'
```
```bash
python test_unlabeled_images.py
```

2. **Configure student training** in `config_Edgenet.py`:
```python
# Data paths
data_dir: str = '/path/to/original/training/images'
gfiqa_data_dir: str = '/path/to/GFIQA/images'
csv_path: str = 'data_file/train.csv'
gfiqa_csv_path: str = 'data_file/gfiqa_results.csv'

# Model save path
model_save_dir: str = '/path/to/save/student/model'
```

3. **Start student training**:
```bash
python train_student_model.py
```

---

## 🧪 Testing

### Pre-trained Models

- **Student Model (Winner)**: [EdgeNeXt-XXS](https://github.com/sunwei925/Efficient-FIQA/releases/download/v1.0/EdgeNeXt_XXS_checkpoint.pt)
- **Teacher Model**: [Swin-B](https://www.dropbox.com/scl/fi/omso4imlippzkmzsq7pzw/Swin_B_checkpoint.pt)
- **Teacher+ Model**: [Swin-B+](https://www.dropbox.com/scl/fi/74abmcgi43t9e9rth012n/Swin_B_plus_checkpoint.pt)


### Test on your images
```bash
python test.py \
    --model_name FIQA_EdgeNeXt_XXS \
    --model_weights_file ckpts/EdgeNeXt_XXS_checkpoint.pt \
    --image_file your_image.jpg \
    --image_size 352 \
    --gpu_ids 0
```

### Usage Examples

```bash
# Test with student model (recommended)
python test.py \
    --model_name FIQA_EdgeNeXt_XXS \
    --model_weights_file ckpts/EdgeNeXt_XXS_checkpoint.pt \
    --image_file demo_images/z06399.png \
    --image_size 352 \
    --gpu_ids 0

# Test with teacher model
python test.py \
    --model_name FIQA_Swin_B \
    --model_weights_file ckpts/Swin_B_plus_checkpoint.pt \
    --image_file demo_images/z06399.png \
    --image_size 448 \
    --gpu_ids 0

# CPU inference
python test.py \
    --model_name FIQA_EdgeNeXt_XXS \
    --model_weights_file ckpts/EdgeNeXt_XXS_checkpoint.pt \
    --image_file demo_images/z06399.png \
    --image_size 352 \
    --gpu_ids cpu
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_name` | Model architecture (`FIQA_EdgeNeXt_XXS` or `FIQA_Swin_B`) | - |
| `--model_weights_file` | Path to model weights | - |
| `--image_size` | Input image size (352 for EdgeNeXt, 448 for Swin-B) | - |
| `--image_file` | Path to input image | - |
| `--gpu_ids` | GPU IDs or "cpu" | "0" |

---

## 🌐 Online Demo

Try our online demo on Hugging Face Spaces: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sunwei925/Efficient-FIQA)

---

## 📚 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{sun2025efficient,
  title={Efficient Face Image Quality Assessment via Self-training and Knowledge Distillation},
  author={Sun, Wei and Zhang, Weixia and Cao, Linhan and Jia, Jun and Zhu, Xiangyang and Zhu, Dandan and Min, Xiongkuo and Zhai, Guangtao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision Workshops},
  pages={1-9},
  year={2025}
}
```

---

<div>

**⭐ Star this repository if you find it helpful!**



</div>