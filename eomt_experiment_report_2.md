# EoMT 실험 2차 보고서 (EoMT ViT-S / ViT-B)

## Throughput 비교

| 항목 | 값 |
|------|----|
| Device | CUDA |
| Input shape | (1, 3, 512, 512) |
| Warm-up | 50 iters |
| Measure | 100 iters |

| 모델 | FPS | Latency |
|------|-----|---------|
| SegFormer-B2 | 65.32 | 15.31 ms/frame |
| EoMT ViT-S | 70.01 | 14.28 ms/frame |
| EoMT ViT-B | 30.02 | 33.31 ms/frame |
| EoMT ViT-L | 12.26 | 81.55 ms/frame |

- EoMT ViT-L은 SegFormer 대비 5배 이상 느려 Hailo 실시간 추론이 현실적으로 어려울 것으로 판단


---

## 목적

EoMT ViT-L(1차)의 속도 문제(12 FPS)를 해결하기 위해 ViT-S, ViT-B 백본으로 교체.  
Hailo 실시간 추론 가능 여부와 성능 저하 폭을 확인한다.

---

## 변경 사항

### 백본: DINOv2 ViT-S

| 항목 | ViT-L (1차) | ViT-S (2차) |
|------|------------|------------|
| backbone_name | vit_large_patch14_reg4_dinov2 | vit_small_patch14_reg4_dinov2 |
| embed_dim | 1024 | 384 |
| total_blocks | 24 | 12 |
| L1 blocks | 20 (block 0~19) | 8 (block 0~7) |
| L2 blocks | 4 (block 20~23) | 4 (block 8~11) |

patch_size=16 (FlexiViT), num_upscale=2, dynamic_img_size=True는 동일.

---

### Optimizer / Trainer

| 항목 | 값 |
|------|----|
| base_lr | 1e-4 |
| max_steps | 160,000 |

---

### LLRD

llrd_decay=0.8, llrd_l2_exempt=True (1차와 동일)  
L2 블록이 block 8~11로 바뀌면서 L1 블록 수가 8개로 줄어 exponent 범위가 달라진다.

| 파라미터 그룹 | lr |
|---------------|----|
| Non-backbone (q, class_head, mask_head, upscale) | base_lr = 1e-4 |
| L2 blocks (8~11, llrd_l2_exempt=True) | base_lr = 1e-4 |
| L1 block 7 | base_lr × 0.8^4 ≈ 4.10e-5 |
| L1 block 6 | base_lr × 0.8^5 ≈ 3.28e-5 |
| ... | ... |
| L1 block 0 | base_lr × 0.8^11 ≈ 8.59e-6 |
| embed (patch_embed, pos_embed 등) | base_lr × 0.8^12 ≈ 6.87e-6 |
| norm (backbone 최종 LayerNorm) | base_lr = 1e-4 |

---

### Mask Annealing

L2 blocks (ViT-S block 8~11)에만 적용.

| ViT block | annealing 시작 | annealing 종료 |
|-----------|--------------|--------------|
| block 8 | 0 step | 500 step |
| block 9 | 0 step | 1,000 step |
| block 10 | 0 step | 1,500 step |
| block 11 | 0 step | 2,000 step |

*(1차 ViT-L: block 20~23, end_steps [40,000 / 80,000 / 120,000 / 160,000])*

---

### Callbacks

#### ModelCheckpoint (3종 분리)

1차에서는 val/loss_total 기준 단일 checkpoint만 저장했으나, 이번 실험부터 3가지 지표 기준으로 분리.

| Checkpoint | monitor | mode | save_top_k | save_last |
|------------|---------|------|-----------|-----------|
| model_checkpoint_miou | val/mean_iou | max | 5 | true |
| model_checkpoint_loss | val/loss_total | min | 5 | false |
| model_checkpoint_f1 | val/pixel_f1 | max | 5 | false |

#### Early Stopping

| 항목 | ViT-L (1차) | ViT-S (2차) |
|------|------------|------------|
| monitor | val/loss_total | val/mean_iou |
| mode | min | max |
| patience | 20 epoch | 20 epoch |

---

## EoMT ViT-B 실험 조건

ViT-S 실험과 동일한 조건에서 백본과 early stopping patience만 변경.

### 백본: DINOv2 ViT-B

| 항목 | ViT-S | ViT-B |
|------|-------|-------|
| backbone_name | vit_small_patch14_reg4_dinov2 | vit_base_patch14_reg4_dinov2 |
| embed_dim | 384 | 768 |
| total_blocks | 12 | 12 |
| L1 blocks | 8 (block 0~7) | 8 (block 0~7) |
| L2 blocks | 4 (block 8~11) | 4 (block 8~11) |

### Early Stopping

| 항목 | ViT-S | ViT-B |
|------|-------|-------|
| monitor | val/mean_iou | val/mean_iou |
| mode | max | max |
| patience | 20 epoch | **50 epoch** |

---

## 실험 결과

### 정량 결과

| 날짜 | 모델 | mIoU | Pixel Acc | Pixel F1 | Best Epoch | 종료 Epoch | Dataset |
|------|------|------|-----------|----------|------------|-----------|---------|
| 2025-09-17 | SegFormer-B2 | 64.0% | — | 75.0% | — | — | aidall v1.0 |
| 2026-04-13 | EoMT ViT-S | 66.6% | 94.5% | 77.5% | 352 | 372 | aidall v1.0 |
| 2026-04-13 | EoMT ViT-B | 72.0% | 95.7% | 81.7% | 394 | 444 | aidall v1.0 |
| 2026-04-09 | EoMT ViT-L | 74.9% | 95.7% | 83.9% | 265 | 285 | aidall v1.0 |

---

## Hailo8 호환성 검증 (ViT-B)

비호환 연산 3종 검출: `Einsum` 6개, `Expand` 8개, `Where` 4개.

해결: `Einsum` → MatMul + Reshape 교체, ONNX export 시 `masked_attn_enabled=False`로 Mask Annealing 비활성화.  
수정 후 Hailo8 parse level 통과.

---

