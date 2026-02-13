# DDAN: Dual-Dimensional Attention Network for Image-Text Retrieval

Official PyTorch implementation of **DDAN: Dual-Dimensional Attention Network for Image-Text Retrieval** (Pattern Recognition, under review).

## Overview

DDAN is a novel image-text retrieval method that captures fine-grained intra-instance interactions along both spatial and feature dimensions through:
- **Vertical Perception Module**: Models region-wise dependencies within each feature dimension
- **Horizontal Perception Module**: Captures inter-dimensional relationships across regions
- **Differentiated Set Matching (DSM) Strategy**: Generates multiple global embeddings focusing on different aspects of image-text pairs

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

Download the following datasets:
- **Flickr30K**: [Download link](http://shannon.cs.illinois.edu/DenotationGraph/)
- **MS-COCO**: [Download link](https://cocodataset.org/)

Extract bottom-up attention features using [Faster R-CNN](https://github.com/peteanderson80/bottom-up-attention).

## Training

### Flickr30K
```bash
bash train_eval_f30k.sh
```

### MS-COCO
```bash
bash train_eval_coco.sh
```

## Evaluation

Evaluation is performed automatically after training. Results will be saved in the checkpoint directory.

## Model Architecture

The model consists of:
1. **Visual Encoder**: Faster R-CNN bottom-up attention features → DDA module
2. **Text Encoder**: GloVe + Bi-GRU → DDA module
3. **DDA Module**:
   - Vertical Perception: Ranking-based feature enhancement
   - Horizontal Perception: 4 different AGLs for differentiated global features
4. **Similarity**: Smooth-Chamfer similarity for set matching
5. **Loss**: Triplet ranking loss with hard negatives

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 1024 |
| Number of regions | 36 |
| Batch size          | 200 images + captions |
| Learning rate       | 1e-3                  |
| Optimizer           | AdamW                 |
| Training epochs     | 80                    |
| α (SC similarity)   | 16                    |
| Margin              | 0.2                   |

## Results

### Flickr30K (1K test set)

| Method | I→T R@1 | I→T R@5 | I→T R@10 | T→I R@1 | T→I R@5 | T→I R@10 | rSum |
|--------|---------|---------|----------|---------|---------|----------|------|
| DDAN   | 81.8    | 95.3    | 97.8     | 60.2    | 85.6    | 91.3     | 512.0|

### MS-COCO (1K test set)

| Method | I→T R@1 | I→T R@5 | I→T R@10 | T→I R@1 | T→I R@5 | T→I R@10 | rSum |
|--------|---------|---------|----------|---------|---------|----------|------|
| DDAN   | 81.7    | 96.3    | 98.6     | 64.4    | 90.9    | 96.0     | 527.9|

