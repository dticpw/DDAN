# DDAN: Dual-Dimensional Attention Network for Image-Text Retrieval

DDAN is a fine-grained image-text retrieval model built on the Faster R-CNN + Bi-GRU EMB framework. It introduces two complementary modules:

- **DDA (Dual-Dimensional Attention)**: operates at both the region level and the dimension level simultaneously, enabling selective emphasis on discriminative feature dimensions rather than treating all dimensions uniformly.
- **DSM (Dual-dimensional Semantic Module)**: generates structurally diverse multi-embeddings through four channels with distinct architectures and explicit functional roles, without relying on regularization losses.

---

## Architecture

### Image Encoder
- Visual features: Faster R-CNN bottom-up features, top-36 regions, 2048-dim
- Projection: FC (2048 → 1024) + MLP residual
- DSM produces 4 embeddings per image via four AGL channels

### Text Encoder
- Word embedding: GloVe (300-dim)
- Sentence encoder: Bi-GRU → 1024-dim
- DSM produces 4 embeddings per caption via four AGL channels

### DSM Channels (AGL)
| Channel | Name | Architecture | Temperature |
|---------|------|-------------|-------------|
| 1 | Conservative Aggregation | Linear | — |
| 2 | Selective Aggregation | Linear → ReLU → Linear | τ = 0.8 |
| 3 | Comprehensive Aggregation | Linear → LeakyReLU → Linear | τ = 1.5 |
| 4 | Semantic Aggregation | MLP (1024 → 2048 → 1024) | — |

Channel outputs are stacked along the embedding dimension, followed by softmax residual and ℓ₂ normalization.

---

## Performance

### Flickr30K (1K test set)

| Method      | I2T R@1     | I2T R@5     | I2T R@10    | T2I R@1     | T2I R@5     | T2I R@10    | rSum         |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ------------ |
| DDAN (ours) | 81.80 ±0.40 | 95.20 ±0.30 | 97.73 ±0.12 | 60.27 ±0.09 | 85.64 ±0.05 | 91.41 ±0.11 | 512.05 ±0.19 |

### MS-COCO (1K test set)

| Method      | I2T R@1     | I2T R@5     | I2T R@10    | T2I R@1     | T2I R@5     | T2I R@10    | rSum         |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ------------ |
| DDAN (ours) | 81.59 ±0.17 | 96.32 ±0.05 | 98.59 ±0.06 | 64.29 ±0.14 | 90.99 ±0.06 | 96.01 ±0.06 | 527.78 ±0.14 |

### MS-COCO (5K test set)

| Method      | I2T R@1     | I2T R@5     | I2T R@10    | T2I R@1     | T2I R@5     | T2I R@10    | rSum         |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ------------ |
| DDAN (ours) | 61.51 ±0.14 | 86.33 ±0.10 | 92.07 ±0.09 | 42.41 ±0.12 | 72.36 ±0.12 | 82.69 ±0.03 | 437.37 ±0.25 |



## Model Size

| Component | Parameters |
|-----------|-----------|
| DDA + DSM modules | 23.09M |
| Total | 70.53M |

---

## Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| MS-COCO | 123,287 images, 5 captions/image | [COCO 2014](https://cocodataset.org/#download) |
| Flickr30K | 31,783 images, 5 captions/image | [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) |
| Visual Genome | Used for bottom-up feature pre-training | [VG](https://visualgenome.org/api/v0/api_home.html) |



---

## Checkpoints

| Dataset | rSum | Download |
|---------|------|----------|
| Flickr30K | 512.05 | https://huggingface.co/dticpw/DDAN/tree/main |
| MS-COCO 5K | 437.4 | https://huggingface.co/dticpw/DDAN/tree/main |

---

## Data Preparation

```
data/
├── coco_butd/
│   ├── train_ids.txt
│   ├── train_caps.txt
│   └── ...
├── f30k_butd/
│   └── ...
vocab/
├── coco_butd_vocab.pkl
└── f30k_butd_vocab.pkl
```

---

## Training

**MS-COCO:**

```bash
python3 train.py \
  --data_name coco_butd --wemb_type glove \
  --data_path /path/to/dataset/ \
  --margin 0.2 --max_violation \
  --img_num_embeds 4 --txt_num_embeds 4 \
  --img_attention --txt_attention --img_finetune --txt_finetune \
  --batch_size 200 --num_epochs 80 \
  --optimizer adamw --lr_scheduler cosine \
  --lr 1e-3 --weight_decay 1e-4 --grad_clip 1 \
  --loss smooth_chamfer --eval_similarity smooth_chamfer --temperature 16 \
  --arch slot --tau_selective 0.8 --tau_comprehensive 1.5 \
  --dropout 0.1 --seed 1 --amp --eval_on_gpu
```

**Flickr30K:**

```bash
python3 train.py \
  --data_name f30k_butd --wemb_type glove \
  --data_path /path/to/dataset/ \
  --margin 0.2 --max_violation \
  --img_num_embeds 4 --txt_num_embeds 4 \
  --img_attention --txt_attention --img_finetune --txt_finetune \
  --batch_size 128 --num_epochs 80 \
  --optimizer adamw --lr_scheduler cosine \
  --lr 1e-3 --weight_decay 1e-4 --grad_clip 1 \
  --loss smooth_chamfer --eval_similarity smooth_chamfer --temperature 16 \
  --arch slot --tau_selective 0.8 --tau_comprehensive 1.5 \
  --dropout 0.1 --seed 1 --amp --eval_on_gpu
```

Or use the provided scripts:

```bash
bash train_eval_coco.sh
bash train_eval_f30k.sh
```

---

## Evaluation

```bash
python3 eval.py \
  --data_name coco_butd \
  --data_path /path/to/dataset/ \
  --ckpt /path/to/checkpoint.pth \
  --eval_on_gpu --eval_similarity smooth_chamfer --temperature 16
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embed_size` | 1024 | Joint embedding dimension |
| `word_dim` | 300 | GloVe word embedding dimension |
| `img_num_embeds` | 4 | Number of image embeddings (DSM channels) |
| `txt_num_embeds` | 4 | Number of text embeddings (DSM channels) |
| `tau_selective` | 0.8 | Temperature for Channel 2 (selective) |
| `tau_comprehensive` | 1.5 | Temperature for Channel 3 (comprehensive) |
| `temperature` | 16 | Smooth-Chamfer similarity temperature |
| `batch_size` | 128 | size of mini-batch |
| `lr` | 1e-3 | Initial learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient clipping max norm |
| `dropout` | 0.1 | Dropout rate |
| `max_seq_len` | 68 | Maximum caption token length |
| `vocab_size` | 8,482 | Vocabulary size (COCO training set) |
| `num_regions` | 36 | Top-K Faster R-CNN regions per image |



