# Efficient-FIQA
![visitors](https://visitor-badge.laobi.icu/badge?page_id=sunwei925/Efficient-FIQA) [![](https://img.shields.io/github/stars/sunwei925/Efficient-FIQA)](https://github.com/sunwei925/Efficient-FIQA)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.13%2B-brightgree?logo=PyTorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/UIQA)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sunwei925/Efficient-FIQA)
[![arXiv](https://img.shields.io/badge/build-paper-red?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2507.15709)

🏆 🥇 **Winner solution for [ICCV VQualA 2025 Face Image Quality Assessment Challenge](https://codalab.lisn.upsaclay.fr/competitions/23017) at the [VQualA 2025](https://vquala.github.io/) workshop @ ICCV 2025** 

Official Code for **Efficient Face Image Quality Assessment via Self-training and Knowledge Distillation**

## Introduction
> **Face image quality assessment (FIQA)** is essential for various face-related applications. Although FIQA has been extensively studied and achieved significant progress, the computational complexity of FIQA algorithms remains a key concern for ensuring scalability and practical deployment in real-world systems. In this paper, we aim to develop a computationally efficient FIQA method that can be easily deployed in real-world applications. Specifically, our method consists of two stages: **training a powerful teacher model** and **distilling a lightweight student model from it**. To build a strong teacher model, we adopt a **self-training strategy** to improve its capacity. We first *train the teacher model using labeled face images, then use it to generate pseudo-labels for a set of unlabeled images*. These pseudo-labeled samples are used in two ways: *(1) to distill knowledge into the student model, and (2) to combine with the original labeled images to further enhance the teacher model through self-training*. The enhanced teacher model is used to further pseudo-label another set of unlabeled images for distilling the student models. The student model is trained using *a combination of labeled images, pseudo-labeled images from the original teacher model, and pseudo-labeled images from the enhanced teacher model*. Experimental results demonstrate that our student model achieves comparable performance to the teacher model with an extremely low computational overhead. Moreover, **our method achieved first place in the ICCV VQualA 2025 FIQA Challenge.**

#### Performance on ICCV VQualA 2025 FIQA Challenge
| **Rank** | **Team**                   | **Score** | **GFLOPs** | **Params (M)** |
|:--------:|----------------------------|:---------:|:----------:|:--------------:|
| 1        | **ECNU-SJTU VQA Team (Ours)**  | **0.9664**    | **0.3313**     | **1.1796**         |
| 2        | MediaForensics             | 0.9624    | 0.4687     | 1.5189         |
| 3        | Next                       | 0.9583    | 0.4533     | 1.2224         |
| 4        | ATHENAFace                 | 0.9566    | 0.4985     | 2.0916         |
| 5        | NJUPT-IQA-Group            | 0.9547    | 0.4860     | 3.7171         |
| 6        | ECNU VIS Lab               | 0.9406    | 0.4923     | 3.2805         |

**Table:** Top-6 challenge results on the FIQA track, ranked by the official final score. The score is computed as the average of SRCC and PLCC.


- for more results on the ICCV VQualA 2025 FIQA Challenge, please refer to the challenge report.

## Usage
### Environments
- Requirements:
```
torch(>=1.13), timm
```
- Create a new environment
```
conda create -n EfficientFIQA python=3.9
conda activate EfficientFIQA 
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia # this command install pytorch version of 2.40, you can install pytorch >=1.13
pip install timm
```


### Dataset
Download the [Training dataset](https://drive.google.com/file/d/1FpylY9uVOfdKw5vI6UduMviUMIiDfK-7/view), [Ground truth scores](https://drive.google.com/file/d/1UQ8m4gIPg5X2LC3ugifWGhul0LP86_9b/view), and [Validation dataset](https://drive.google.com/file/d/1UM8IgjFjf6O3hIwhVqfFaLkHMjtMOvK2/view).

### Training
The training code will be added.

### Testing
The student model (EdgeNeXt) is avaliable at ckpts folder. (**The winner model for this challenge**.)

The teacher model (Swin-B) can be download from [Dropbox](https://www.dropbox.com/scl/fi/omso4imlippzkmzsq7pzw/Swin_B_checkpoint.pt?rlkey=fjcdrnnnm8uipahsg0d4ej4zs&st=rclvat38&dl=0), [Baidu Yun](https://pan.baidu.com/s/1bF2k0z0ZmPRQJ3YlO-AEGQ) (ap96)

The teacher plus model (Swin-B) can be download from [Dropbox](https://www.dropbox.com/scl/fi/74abmcgi43t9e9rth012n/Swin_B_plus_checkpoint.pt?rlkey=m7whu8j4yvtkoiu3nscivfkkd&st=i3wia01p&dl=0), [Baidu Yun](https://pan.baidu.com/s/1z0iZw3T1X3Ul11WCzwbd9w) (qpiv).

To test a trained model on your own images, use the `test.py` script:

```bash
python test.py [options]
```

Available options:
- `--model_name`: Model architecture to use (choices: 'FIQA_EdgeNeXt_XXS' or 'FIQA_Swin_B')
- `--model_weights_file`: Path to model weights file
- `--image_size`: Input image size (352 for FIQA_EdgeNeXt_XXS and 448 for FIQA_Swin_B)
- `--image_file`: Path to input image file 
- `--gpu_ids`: GPU IDs to use (e.g., "0", "0,1", or "cpu" for CPU only)

Example usage:
```bash
# Test with EdgeNeXt-XXS model on GPU
python test.py \
--model_name FIQA_EdgeNeXt_XXS \
--model_weights_file ckpts/EdgeNeXt_XXS_checkpoint.pt \
--image_file demo_images/z06399.png \
--image_size 352 \
--gpu_ids 0

# Test with EdgeNeXt-XXS model on CPU
python test.py \
--model_name FIQA_EdgeNeXt_XXS \
--model_weights_file ckpts/EdgeNeXt_XXS_checkpoint.pt \
--image_file demo_images/z06399.png \
--image_size 352 \
--gpu_ids cpu

# Test with Swin_B model on GPU
python test.py \
--model_name FIQA_Swin_B \
--model_weights_file ckpts/Swin_B_plus_checkpoint.pt \
--image_file demo_images/z06399.png \
--image_size 448 \
--gpu_ids 0
```

The script will output the quality score for the input image, with higher scores indicating better image quality.

### Online Demo
Try our online demo on Hugging Face Spaces: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sunwei925/Efficient-FIQA)

