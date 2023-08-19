## MECOM：A Meta-Completion Network for Fine-Grained Recognition with Incomplete Multi-Modalities

This repository is the official PyTorch implementation of MECOM：A Meta-Completion Network for Fine-Grained Recognition with Incomplete Multi-Modalities, which provides practical and effective method for fine-grained recognition with modality missing.

## Main requirements
```bash
torch >= 1.4.0
cuda >=0.10.0
torchvision >= 0.9.0
Python >=3
timm
```
- These versions are only recommended and not mandatory

## Preparing the datasets

We provide three datasets in this repo: UPMC Food-101, PKU FG-XMedia, RGB-D. The first half of each dataset categorys are divided into meta train sets and others are divided into meta test sets.

The detailed information of these datasets are shown as follows:

| CATEGORY    | UPMC Food-101 | PKU FG-XMedia | RGB-D |
|-------------|---------------|---------------|-------|
| Total       | 101           | 200           | 51    |
| In Dmeta_tr | 50            | 100           | 25    |
| In Dmeta_te | 51            | 100           | 26    |

As mentioned in MECOM, we use different methods to preprocess each modality data and store them in "pth" format. For example, in PKU FG-XMedia meta test set, there are 2930 images and they would be stroe as "PKU_image_test.pth" which is a tensor with size of (2930,2048).

# Usage

### Data Processing

Take image modality for example, we firstly create feature extrcation model from timm (ResNet50) and train the model with image labels. 
```bash
import timm
model = timm.create_model('ResNet50', pretrained=True, num_classes=100)
model.train()
# load data and train
..
```
Secondly we use the trained model to extract image features and store them in "pth" format.
```bash
model = timm.create_model('resnet50', pretrained=False, num_classes=100)
# load trained model
..
model.reset_classifier(0)
# feature extraction
..
```

### Training
There are two data missing patterns in MECOM, one is block wise missing and the other is element wise missing. Both missing patterns are included in ./util/get_mask.py. Take block wise missing for example:
```bash
# train model in meta train set
python meta_learn.py
```

### Validation
```bash
# val model in meta test set
python block_wise_main.py
```

