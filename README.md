# DRIVE: Dynamic Relation Inference via Verb Embeddings

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

**DRIVE** (Dynamic Relation Inference via Verb Embeddings) is a novel approach to enhance relation detection in images. By augmenting the MS COCO dataset with subject-relation-object annotations and fine-tuning CLIP (Contrastive Language-Image Pre-training) models using weakly contrastive triples, DRIVE introduces a unique method and loss function that significantly improves zero-shot relation inference accuracy. Our method outperforms CLIP and state-of-the-art models in both frozen and fine-tuned settings.

## Prerequisites

- **Python 3.x**
- **PyTorch**
- **CUDA-compatible GPU** (for training)
- **MS COCO 2017 Dataset**
- Additional libraries (install necessary libraries based on the import statements in the code)

## Installation
   ```bash
   git clone https://github.com/ai-lab-conf/Dynamic-Relation-Inference-via-Verb-Embeddings.git
   cd DRIVE
   ```

## Dataset Preparation

### Downloading CROCO and CROCO_D Datasets

Run the following script from the root directory to download and set up the necessary datasets:

```bash
bash scripts/fill_dataset.sh
```

This script will:

- Unzip dataset annotations.
- Download images from the MS COCO 2017 dataset and generate the CRODO dataset.
- Download images from the CROCO_D dataset.
- Organize images into appropriate splits for each dataset.

**Note:** Ensure you have sufficient disk space (datasets can be large).

## Training the Model

### Running the Training Pipeline

Execute the training script:

```bash
bash scripts/train_script.sh
```

This script uses predefined paths and parameters. Adjust the script if you need to modify training configurations.

### Training Configuration

Default parameters:

- **Batch Size**: 64
- **Epochs**: 7
- **Learning Rate**: 1.8748e-05
- **Weight Decay**: 4.5269e-05
- **Warmup Steps**: 1,430
- **Precision**: Mixed Precision (AMP)
- **Gradient Clipping Norm**: 1.0
- **Delta_i (δᵢ)**: 1.2233
- **Delta_t (δₜ)**: 0.615

#### Customizing Training Parameters

To modify training parameters:

- **Edit** `scripts/train_script.sh` with your desired configurations.
- **Directly pass arguments** when running the training script:

  ```bash
  python train.py --batch_size 128 --epochs 10 --learning_rate 2e-5
  ```

## Evaluation

After training, evaluation begins automatically if the provided script is used. 
The evaluation script displays metrics for relation inference accuracy of all the relevant models.
