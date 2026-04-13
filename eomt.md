# EoMT 구현 분석

논문: [Your ViT is Secretly an Image Segmentation Model (2025)](https://arxiv.org/abs/2503.19108)

---

## 파일 구조

```
src/aidall_seg/
├── lr_scheduler.py               ← TwoStageWarmupPolySchedule 포함
├── loss/
│   ├── __init__.py               ← EoMTLoss export
│   └── eomt_loss.py              ← Hungarian matching + CE + BCE + Dice
└── models/
    ├── backbones/
    │   ├── __init__.py           ← PlainViTBackbone export
    │   └── plain_vit.py          ← timm ViT 범용 래퍼 (DINOv2 등 로드)
    ├── __init__.py               ← EoMT, EoMTLightningModule export
    ├── eomt.py                   ← EoMT 모델 아키텍처
    └── base.py                   ← EoMTLightningModule (학습 루프, optimizer)

configs/
├── model/eomt/
│   ├── base.yaml                 ← num_q, poly_power, loss 계수 공통 설정
│   └── eomt-vitl.yaml            ← ViT-L backbone 설정
├── optimizer/eomt/
│   └── adamw.yaml                ← lr, weight_decay, llrd_decay, llrd_l2_exempt
├── augmentation/
│   └── eomt_aug.yaml             ← Resize/Crop/Flip/Normalize
└── experiment/eomt/
    └── eomt-vitl.yaml            ← 전체 통합 실험 설정
```

---

## 모듈별 기능

### `plain_vit.py` — PlainViTBackbone

timm을 통해 DINOv2 ViT-L 가중치를 로드하는 백본 래퍼.

- `__init__`: timm으로 ViT 로드, `l2_start = total_blocks - l2_blocks` 계산 (ViT-L: 24 - 4 = 20)
- `forward_l1`: 이미지 → 패치 임베딩 → 위치 임베딩 → L1 블록(0~19) 통과
- `l2_blocks` (property): `backbone.blocks[l2_start:]` 참조 반환 (파라미터 이중 등록 방지)
- `norm` (property): 백본 최종 LayerNorm 노출
- `num_upscale`: `log2(patch_size) - 2` 로 계산 (patch_size=16 → num_upscale=2)

### `eomt.py` — EoMT

메인 모델 아키텍처. decoder 없이 encoder만 사용.

**구성 모듈:**
- `self.q`: 학습 가능한 쿼리 토큰 `(num_q, embed_dim)` — 기본 100개
- `self.class_head`: Linear `(embed_dim → num_classes + 1)` — +1은 no-object 클래스
- `self.mask_head`: Linear 3층 MLP `(embed_dim → embed_dim)`
- `self.upscale`: ScaleBlock × num_upscale — 패치 해상도를 2배씩 복원
- `self.attn_mask_probs`: L2 블록 수 크기의 buffer — 각 블록의 masked attention 적용 확률

**ScaleBlock:**
ConvTranspose2d(2x upsample) → GELU → Depthwise Conv2d → LayerNorm2d

**forward 흐름:**
```
입력 이미지 (B, 3, H, W)
    ↓ forward_l1
L1 블록 출력 (B, N_patches + prefix_tokens, embed_dim)
    ↓ query concat
(B, num_q + N_patches + prefix_tokens, embed_dim)
    ↓ L2 블록 순회 (4회)
        각 블록 진입 전:
          - _predict(): normed_x → class_logits, mask_logits 계산
          - _attn_mask(): mask_logits로 boolean attention mask 생성
          - _disable_attn_mask(): attn_mask_probs[i] 확률로 일부 마스크 해제 (annealing)
        블록 내부: norm1 → masked self-attention → residual → norm2 → MLP → residual
    ↓ 최종 LayerNorm 후 _predict()
mask_logits_per_layer (L2 블록 수 + 1), class_logits_per_layer 반환
```

**_predict():**
- 쿼리 토큰 `q = x[:, :num_q]` 에서 class_logits 계산
- 패치 토큰을 2D로 reshape → upscale → mask_head(q)와 einsum → mask_logits

**_attn_mask():**
- `(B, N, N)` boolean mask 생성 (전체 True로 초기화)
- 쿼리가 패치를 볼 수 있는 위치만 `mask_logits > 0`으로 제한
- `_disable_attn_mask()`로 `attn_mask_probs[i]` 확률만큼 마스크 해제

### `eomt_loss.py` — EoMTLoss

HuggingFace `Mask2FormerLoss`를 상속. Hungarian Matching으로 쿼리-GT 매칭 후 loss 계산.

- **Hungarian Matching**: 각 쿼리와 GT 마스크 간 최적 1:1 매칭 (매 배치마다 수행)
- **loss_masks**: Point Sampling 기반 BCE + Dice loss
- **loss_labels**: matched 쿼리 CE loss (no-object 가중치 0.1)
- **DDP 동기화**: `dist.all_reduce`로 전체 GPU 마스크 수 합산 후 정규화

반환: `{"loss_mask": t1, "loss_dice": t2, "loss_cross_entropy": t3}`

### `base.py` — EoMTLightningModule

`SegmentationLightningModule` 상속. EoMT 전용 학습 루프.

**`__init__` 파라미터:**
- `poly_power`: mask annealing + LR poly decay 곡선 조절 (기본 0.9)
- `warmup_steps`: `[non_backbone_warmup, backbone_warmup]` — TwoStageWarmupPolySchedule에 전달
- `attn_mask_annealing_start_steps`: 블록별 annealing 시작 스텝 리스트
- `attn_mask_annealing_end_steps`: 블록별 annealing 종료 스텝 리스트

**`training_step`:**
1. `forward()` → `mask_logits_per_layer`, `class_logits_per_layer`
2. 마지막 레이어로 main loss 계산
3. 나머지 레이어로 auxiliary loss 계산 (각각 `_{k}_aux_{i}` key로 로깅)
4. `_calculate_total_loss()`: 계수(class×2, mask×5, dice×5) 적용 합산
5. `_fuse_to_semantic_logits()`: `softmax(class) × sigmoid(mask)` einsum → semantic logits
6. metric 업데이트 및 로깅

**`on_train_batch_end` — Mask Annealing:**
```
블록별 start/end step이 설정된 경우:
  각 블록 i:
    step < start_steps[i]  → prob = 1.0 (완전 masked)
    step >= end_steps[i]   → prob = 0.0 (마스크 완전 해제)
    그 사이                → prob = (1 - progress)^poly_power (poly decay)
  model.attn_mask_probs[i] = prob

fallback (start/end 미설정):
  전체 블록 동시에 uniform poly decay
```

**`configure_optimizers` — LLRD + TwoStageWarmupPolySchedule:**
```
파라미터 그룹 구성:
  [0] decoder_params       lr = base_lr          (q, class_head, mask_head, upscale)
  [1] backbone block 0     lr = base_lr × 0.8^23 (L1, 가장 앞)
  ...
  [20] backbone block 20   lr = base_lr           (L2, llrd_l2_exempt=True면 복원)
  [21] backbone block 21   lr = base_lr
  [22] backbone block 22   lr = base_lr
  [23] backbone block 23   lr = base_lr
  [24] embed_params         lr = base_lr × 0.8^24 (patch_embed, pos_embed 등)
  [25] norm_params          lr = base_lr           (backbone 최종 LayerNorm)

num_non_backbone_groups = 1 → TwoStageWarmupPolySchedule에 전달
```

### `lr_scheduler.py` — TwoStageWarmupPolySchedule

| 구간 | Non-backbone (index < 1) | Backbone (index >= 1) |
|------|--------------------------|----------------------|
| `0 ~ non_bb_w` | `base_lr × (step / non_bb_w)` | `0.0` (frozen) |
| `non_bb_w ~ non_bb_w + bb_w` | poly decay | `base_lr × (step - non_bb_w) / bb_w` |
| `non_bb_w + bb_w ~ total` | poly decay | poly decay |

poly decay: `base_lr × max(0, (1 - adjusted / max_steps)^poly_power)`

---

## 설정 파일

### `configs/model/eomt/base.yaml`
```yaml
_target_: aidall_seg.models.EoMTLightningModule
poly_power: 0.9
model:
  _target_: aidall_seg.models.EoMT
  num_q: 100
criterion:
  _target_: aidall_seg.loss.EoMTLoss
  class_coefficient: 2.0
  mask_coefficient: 5.0
  dice_coefficient: 5.0
  no_object_coefficient: 0.1
```

### `configs/model/eomt/eomt-vitl.yaml`
```yaml
model:
  backbone:
    _target_: aidall_seg.models.backbones.PlainViTBackbone
    backbone_name: vit_large_patch14_reg4_dinov2
    img_size: 518
    l2_blocks: 4
```

### `configs/optimizer/eomt/adamw.yaml`
```yaml
_target_: torch.optim.AdamW
_partial_: true
lr: 1e-4
weight_decay: 0.05
llrd_decay: 0.8
llrd_l2_exempt: true
betas: [0.9, 0.999]
eps: 1e-8
```

### `configs/experiment/eomt/eomt-vitl.yaml` (주요 설정)
```yaml
model:
  num_classes: 18
  warmup_steps: [500, 1000]
  attn_mask_annealing_start_steps: [0, 0, 0, 0]
  attn_mask_annealing_end_steps: [40000, 80000, 120000, 160000]
trainer:
  max_steps: 160000
  accumulate_grad_batches: 32   # effective batch size = 2 × 32 = 64
```

---

## 전체 학습 플로우

```
Hydra 설정 로드
    ↓
EoMTLightningModule 생성
  └─ EoMT 생성
       └─ PlainViTBackbone (timm DINOv2 ViT-L 가중치 로드)
  └─ EoMTLoss 생성 (Hungarian Matcher 포함)

configure_optimizers()
  ├─ LLRD 파라미터 그룹 구성 (decoder / L1 blocks / L2 blocks / embed / norm)
  └─ TwoStageWarmupPolySchedule 생성

학습 루프 (매 step):
  training_step()
    ├─ forward(): L1 → query concat → L2(masked attn) → 예측
    ├─ EoMTLoss: Hungarian matching → CE + BCE + Dice
    └─ semantic logits로 mIoU 계산

  on_train_batch_end()
    └─ 블록별 attn_mask_probs 업데이트 (poly annealing)

  LRScheduler.step() (interval="step")
    └─ TwoStageWarmupPolySchedule: warmup → poly decay
```

---

## Mask Annealing 스케줄 (ViT-L, 160k steps)

| 블록 | annealing 시작 | annealing 종료 | 의미 |
|------|---------------|---------------|------|
| L2 block 0 (block 20) | 0 | 40,000  | 가장 먼저 마스크 해제 |
| L2 block 1 (block 21) | 0 | 80,000  | |
| L2 block 2 (block 22) | 0 | 120,000 | |
| L2 block 3 (block 23) | 0 | 160,000 | 마지막까지 masked attention 유지 |
