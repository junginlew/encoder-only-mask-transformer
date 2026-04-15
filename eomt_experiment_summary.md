# EoMT Experiment

## 목적
SegFormer의 decoder 병목을 해결하기 위해 EoMT 모델로의 대체 가능성을 확인한다.
Dataset: aidall v1.0

## Backbone (DINOv2 ViT-L)

| Hyperparameter | Value |
|----------------|-------|
| backbone | vit_large_patch14_reg4_dinov2 |
| pretrained weights | DINOv2 (timm) |
| embed_dim | 1024 |
| total_blocks | 24 (L1: block 0~19, L2: block 20~23) |
| patch_size | 16 (FlexiViT: 14→16 interpolation) |
| input_size | 512×512 |


## Training (ViT-L)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| base_lr | 1e-4 |
| weight_decay | 0.05 |
| LLRD decay | 0.8 (L2 blocks exempt → base_lr) |
| LR scheduler | TwoStageWarmupPoly (non-backbone warmup 0~500, backbone 500~1500, poly decay after) |
| max_steps | 160,000 |
| batch_size | 2 × grad_accum 32 = effective 64 |
| Loss | EoMTLoss (class ×2, mask ×5, dice ×5) + auxiliary ×4 |
| Mask Annealing end_steps | block 20: 40K / block 21: 80K / block 22: 120K / block 23: 160K |
| Augmentation | RandomCrop, HorizontalFlip, RandomBrightnessContrast, HueSaturationValue |
| Early Stopping | patience 20 epochs, monitor val/mean_iou |
| Best epoch | 265 / 285 (stopped) |

---

## Backbone (DINOv2 ViT-B)

| Hyperparameter | Value |
|----------------|-------|
| backbone | vit_base_patch14_reg4_dinov2 |
| pretrained weights | DINOv2 (timm) |
| embed_dim | 768 |
| total_blocks | 12 (L1: block 0~7, L2: block 8~11) |
| patch_size | 16 (FlexiViT: 14→16 interpolation) |
| input_size | 512×512 |


## Training (ViT-B)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| base_lr | 1e-4 |
| weight_decay | 0.05 |
| LLRD decay | 0.8 (L2 blocks exempt → base_lr) |
| LR scheduler | TwoStageWarmupPoly (non-backbone warmup 0~500, backbone 500~1500, poly decay after) |
| max_steps | 160,000 |
| batch_size | 2 × grad_accum 32 = effective 64 |
| Loss | EoMTLoss (class ×2, mask ×5, dice ×5) + auxiliary ×4 |
| Mask Annealing end_steps | block 8: 500 / block 9: 1,000 / block 10: 1,500 / block 11: 2,000 |
| Augmentation | RandomCrop, HorizontalFlip, RandomBrightnessContrast, HueSaturationValue |
| Early Stopping | patience 50 epochs, monitor val/mean_iou |
| Best epoch | 394 / 444 (stopped) |

---

## Backbone (DINOv2 ViT-S)

| Hyperparameter | Value |
|----------------|-------|
| backbone | vit_small_patch14_reg4_dinov2 |
| pretrained weights | DINOv2 (timm) |
| embed_dim | 384 |
| total_blocks | 12 (L1: block 0~7, L2: block 8~11) |
| patch_size | 16 (FlexiViT: 14→16 interpolation) |
| input_size | 512×512 |


## Training (ViT-S)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| base_lr | 1e-4 |
| weight_decay | 0.05 |
| LLRD decay | 0.8 (L2 blocks exempt → base_lr) |
| LR scheduler | TwoStageWarmupPoly (non-backbone warmup 0~500, backbone 500~1500, poly decay after) |
| max_steps | 160,000 |
| batch_size | 2 × grad_accum 32 = effective 64 |
| Loss | EoMTLoss (class ×2, mask ×5, dice ×5) + auxiliary ×4 |
| Mask Annealing end_steps | block 8: 500 / block 9: 1,000 / block 10: 1,500 / block 11: 2,000 |
| Augmentation | RandomCrop, HorizontalFlip, RandomBrightnessContrast, HueSaturationValue |
| Early Stopping | patience 20 epochs, monitor val/mean_iou |
| Best epoch | 352 / 372 (stopped) |

---

## Results

| Model | mIoU | Pixel Acc | Pixel F1 | Best Epoch | 종료 Epoch |
|------|------|-----------|----------|------------|-----------|
| SegFormer-B2 | 64.0% | — | 75.0% | — | — |
| EoMT ViT-S | 66.6% | 94.5% | 77.5% | 352 | 372 |
| EoMT ViT-B | 72.0% | 95.7% | 81.7% | 394 | 444 |
| EoMT ViT-L | 74.9% | 95.7% | 83.9% | 265 | 285 |

---

## Throughput

| Model | FPS | Latency |
|------|-----|---------|
| SegFormer-B2 | 65.32 | 15.31 ms/frame |
| EoMT ViT-S | 70.01 | 14.28 ms/frame |
| EoMT ViT-B | 30.02 | 33.31 ms/frame |
| EoMT ViT-L | 12.26 | 81.55 ms/frame |

## Hailo8 호환성 검증 (ViT-B)

비호환 연산 3종 검출: `Einsum` 6개, `Expand` 8개, `Where` 4개.

해결: `Einsum` → MatMul + Reshape 교체, ONNX export 시 `masked_attn_enabled=False`로 Mask Annealing 비활성화.  
수정 후 Hailo8 parse level 통과.
