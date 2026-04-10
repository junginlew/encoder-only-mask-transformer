# EoMT 실험 1차 보고서

## 실험 요약

| 날짜 | 모델 | mIoU | F1-Score | Dataset |
|------|------|------|----------|---------|
| 2026-04-09 | EoMT ViT-L | **74.9%** | **83.9%** | aidall v1.0 |
| 2025-09-17 | SegFormer-B2 *(baseline)* | 64% | 75% | aidall v1.0 |

---

## 목적

기존 SegFormer-B2 대비 EoMT ViT-L의 성능을 동일 데이터셋(aidall v1.0)에서 비교한다.  
EoMT는 별도의 decoder 없이 encoder 내부에서 mask-classification을 수행하는 구조로,  
DINOv2 ViT-L의 사전학습 표현을 최대한 보존하면서 세그멘테이션에 특화된 학습 전략을 적용한다.

---

## 모델 아키텍처

### 백본: PlainViTBackbone (DINOv2 ViT-L)

| 항목 | 값 |
|------|----|
| backbone_name | vit_large_patch14_reg4_dinov2 |
| img_size | 518 (초기화 기준, runtime은 dynamic) |
| patch_size | 16 (FlexiViT: DINOv2 14→16 보간) |
| embed_dim | 1024 |
| total_blocks | 24 |
| l2_blocks | 4 (block 20~23) |
| l1_blocks | 20 (block 0~19) |
| num_upscale | 2 (H/16 → H/8 → H/4) |
| pretrained | DINOv2 사전학습 가중치 (timm 자동 로드) |

> patch_size=16 사용 이유: 실제 학습/추론 입력은 512×512 (augmentation crop). 512/16=32 → 32×32 패치 grid → ScaleBlock 2회 업스케일 → 128×128 (H/4) 출력. patch_size=14를 쓰면 512/14≈36.6으로 나누어 떨어지지 않아 해상도 산술이 불명확해지므로, FlexiViT 기법으로 DINOv2 patch_embed 가중치를 14→16으로 보간해 사용.

### EoMT 헤드

| 항목 | 값 |
|------|----|
| num_q (쿼리 수) | 100 |
| class_head | Linear (1024 → num_classes + 1) |
| mask_head | Linear MLP 3층 (1024 → 1024) |
| upscale | ScaleBlock × 2 (ConvTranspose2d + Depthwise Conv) |
| masked_attn_enabled | True |

### Forward 흐름

```
입력 (B, 3, 512, 512)
  ↓ L1 블록 (block 0~19): 패치화 + 위치 임베딩 + 20개 블록 통과
  ↓ Query concat: (B, 100 + N_patches + prefix, 1024)
  ↓ L2 블록 (block 20~23) × 4회:
      각 블록 진입 전 → class_logits, mask_logits 예측
                      → mask_logits 기반 boolean attention mask 생성
                      → attn_mask_probs[i] 확률로 마스크 일부 해제
      블록 내부 → Masked Self-Attention → MLP
  ↓ 최종 LayerNorm 후 예측
출력: mask_logits_per_layer (5개), class_logits_per_layer (5개)
  ↓ Semantic Logit Fusion
      softmax(class_logits)[..., :-1] × sigmoid(mask_logits) → einsum → (B, C, H/4, W/4)
  ↓ Bilinear interpolate → (B, C, H, W)
```

> L2 블록마다 중간 예측을 수행하고, 마지막 예측 외 나머지는 auxiliary loss로 학습에 활용.

---

## 데이터셋

| 항목 | 값 |
|------|----|
| 데이터셋 | aidall v1.0 |
| Train | 402장 (AIDALL 312 + AIHUB 90) |
| Validation | 74장 (AIDALL 60 + AIHUB 14) |
| num_classes | 18 (class 0~17) |
| ignore_index | 255 |
| 출처 | AidALL 자체 수집 + AIHUB |

### 클래스 정의

| ID | 클래스 | ID | 클래스 |
|----|--------|----|--------|
| 0 | background | 9 | curb |
| 1 | sidewalk | 10 | manhole |
| 2 | crosswalk | 11 | drain |
| 3 | curb-cut | 12 | cautious-zone |
| 4 | terrain | 13 | person |
| 5 | bike-lane | 14 | micromobility |
| 6 | minor-road | 15 | motorcycle |
| 7 | parking-area | 16 | car-four-wheeled |
| 8 | road | 17 | other-vehicle |

---

## 학습 설정

### 데이터 증강

**Train**

| 순서 | 변환 | 파라미터 |
|------|------|---------|
| 1 | SmallestMaxSize | max_size=512 |
| 2 | RandomCrop | 512×512, pad_if_needed=True, fill_mask=255 |
| 3 | HorizontalFlip | p=0.5 |
| 4 | RandomBrightnessContrast | brightness/contrast ±0.2, p=0.8 |
| 5 | HueSaturationValue | hue±20, sat±30, val±20, p=0.3 |
| 6 | Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| 7 | ToTensorV2 | - |

**Validation**: SmallestMaxSize(512) → CenterCrop(512×512) → Normalize → ToTensorV2

---

### Optimizer

| 항목 | 값 |
|------|----|
| Optimizer | AdamW |
| base_lr | 1e-4 |
| weight_decay | 0.05 |
| betas | (0.9, 0.999) |
| eps | 1e-8 |

### LLRD (Layer-wise LR Decay)

백본의 앞쪽 블록일수록 낮은 lr을 적용해 사전학습 표현을 보존한다.  
L2 블록(block 20~23)은 query와 함께 새로운 역할을 수행하므로 LLRD 없이 base_lr 복원.

| 파라미터 그룹 | lr |
|---------------|----|
| Non-backbone (q, class_head, mask_head, upscale) | base_lr = 1e-4 |
| L2 blocks (20~23, llrd_l2_enabled=True) | base_lr = 1e-4 |
| L1 block 19 | base_lr × 0.8^4 |
| L1 block 18 | base_lr × 0.8^5 |
| ... | ... |
| L1 block 0 | base_lr × 0.8^23 ≈ 5.9e-7 |
| embed (patch_embed, pos_embed 등) | base_lr × 0.8^24 ≈ 4.7e-7 |
| norm (backbone 최종 LayerNorm) | base_lr = 1e-4 |

### LR Scheduler: TwoStageWarmupPolySchedule

백본과 non-backbone의 학습 시작 시점을 분리한다.

| 구간 | Non-backbone (decoder) | Backbone |
|------|----------------------|----------|
| 0 ~ 500 step | 0 → base_lr 선형 warmup | frozen (lr=0) |
| 500 ~ 1500 step | poly decay | 0 → base_lr 선형 warmup |
| 1500 ~ 160000 step | poly decay | poly decay |

- poly decay 수식: `lr × max(0, (1 - adjusted/max_steps)^0.9)`
- poly_power: 0.9

### Trainer

| 항목 | 값 |
|------|----|
| max_steps | 160,000 |
| train_batch_size | 2 |
| accumulate_grad_batches | 32 |
| **effective_batch_size** | **64 (2 × 32)** |
| check_val_every_n_epoch | 1 (매 epoch) |
| val_batch_size | 2 |

---

### Loss: EoMTLoss

Hungarian Matching으로 쿼리-GT 마스크를 1:1 매칭한 뒤 loss를 계산한다.

| 항목 | 값 |
|------|----|
| class_coefficient | 2.0 |
| mask_coefficient (BCE) | 5.0 |
| dice_coefficient | 5.0 |
| no_object_coefficient | 0.1 |
| num_points (Point Sampling) | 12,544 |
| oversample_ratio | 3.0 |
| importance_sample_ratio | 0.75 |

**Auxiliary Loss**  
L2 블록 4개 × 중간 예측 + 최종 예측 = 총 5회 loss 계산.  
`total_loss = Σ (class×2 + mask×5 + dice×5)` — main + aux 모두 동일 계수 적용.

---

### Mask Annealing

학습이 진행될수록 masked attention을 서서히 해제한다.  
앞쪽 L2 블록부터 먼저 해제해 초기에는 마스크로 학습을 안정화하고, 후반부엔 자유로운 attention을 허용한다.

| L2 블록 | 해당 ViT block | annealing 시작 | annealing 종료 |
|---------|--------------|--------------|--------------|
| block 0 | block 20 | 0 step | 40,000 step |
| block 1 | block 21 | 0 step | 80,000 step |
| block 2 | block 22 | 0 step | 120,000 step |
| block 3 | block 23 | 0 step | 160,000 step |

- annealing 수식: `prob = max(0, (1 - progress)^poly_power)` (poly_power=0.9)
- prob=1.0: 완전 masked attention / prob=0.0: 마스크 완전 해제

---

## 실험 결과

### 학습 요약

| 항목 | 값 |
|------|----|
| Early Stopping patience | 20 epoch |
| max_steps | 160,000 |
| 재개 checkpoint | epoch_049.ckpt |
| Best epoch | 265 (val/loss_total 기준) |
| 중단 epoch | 285 (20 epoch 개선 없음) |

**Best checkpoint (epoch 265) 기준 지표:**

| 지표 | Train | Val |
|------|-------|-----|
| loss_total | 5.947 | **1.873** |
| mean_iou | 0.879 | **0.749** |
| pixel_acc | - | **0.957** |
| pixel_f1 | - | **0.839** |

### 정량 결과 (Evaluate, epoch 265 checkpoint)

| 모델 | mIoU | Pixel Acc | Pixel F1 |
|------|------|-----------|----------|
| SegFormer-B2 *(baseline)* | 64.0% | - | 75.0% |
| EoMT ViT-L | **74.9%** | **95.7%** | **83.9%** |

- mIoU +10.9%p, Pixel F1 +8.9%p로 성능 우위
- 단, backbone 규모(DINOv2 315M vs MiT-B2 25M) 및 학습 조건이 달라 순수 구조 비교로 보기 어려움

### Throughput (FPS) 비교

| 항목 | 값 |
|------|----|
| Device | CUDA |
| Input shape | (1, 3, 512, 512) |
| Warm-up | 50 iters |
| Measure | 100 iters |

| 모델 | FPS | Latency |
|------|-----|---------|
| SegFormer-B2 | 65.41 | 15.29 ms/frame |
| EoMT ViT-L | 12.20 | 81.94 ms/frame |
| **속도 차이** | **SegFormer 5.36× 빠름** | - |

- SegFormer-B2도 Hailo 가속기에서 상당히 무거운 편
- EoMT ViT-L은 SegFormer 대비 5배 이상 느려 Hailo 실시간 추론이 현실적으로 어려울 것으로 판단

### 클래스별 IoU

*(업데이트 예정)*

### Loss 곡선

*(업데이트 예정)*

---

## 비교 모델 설정 (SegFormer-B2 baseline)

| 항목 | 값 |
|------|----|
| Optimizer | AdamW |
| lr (Decoder) | 2e-4 |
| lr (Encoder) | 1e-5 |
| lr (Classifier) | 2e-3 |
| Steps | 5,000 (warmup 100 steps) |
| LR Scheduler | Poly (power=1.0) |
| weight_decay | 0.01 |
| Pre-training | ImageNet-1K + Mapillary Vistas |
