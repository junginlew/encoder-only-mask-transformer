"""
EoMT 파이프라인 Smoke Test.

DINOv2 사전학습 가중치 다운로드 없이 (pretrained=False) 작은 ViT로
EoMTLightningModule의 training_step / validation_step / configure_optimizers가
오류 없이 동작하는지 검증한다.
"""

from __future__ import annotations

import math
from unittest.mock import patch, PropertyMock

import pytest
import torch
import torch.nn as nn

from aidall_seg.loss.eomt_loss import EoMTLoss
from aidall_seg.models.backbones.plain_vit import PlainViTBackbone
from aidall_seg.models.eomt import EoMT
from aidall_seg.models.base import EoMTLightningModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_CLASSES = 4
IMG_H, IMG_W = 64, 64 
PATCH_SIZE = 16         # 64 / 16 = 4 → 4×4 패치 grid
NUM_Q = 10
BATCH = 2


@pytest.fixture(scope="module")
def tiny_backbone() -> PlainViTBackbone:
    """pretrained=False로 ViT-Tiny 로드 (네트워크 없이 동작)."""
    return PlainViTBackbone(
        backbone_name="vit_tiny_patch16_224",
        img_size=IMG_H,
        patch_size=PATCH_SIZE,
        l2_blocks=2,
        pretrained=False,
    )


@pytest.fixture(scope="module")
def eomt_model(tiny_backbone: PlainViTBackbone) -> EoMT:
    return EoMT(
        backbone=tiny_backbone,
        num_classes=NUM_CLASSES,
        num_q=NUM_Q,
        masked_attn_enabled=True,
    )


@pytest.fixture(scope="module")
def criterion() -> EoMTLoss:
    return EoMTLoss(
        num_classes=NUM_CLASSES,
        num_points=16,   # 포인트 샘플링 수 최소화
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_coefficient=5.0,
        dice_coefficient=5.0,
        class_coefficient=2.0,
        no_object_coefficient=0.1,
    )


@pytest.fixture(scope="module")
def lightning_module(eomt_model: EoMT, criterion: EoMTLoss) -> EoMTLightningModule:
    module = EoMTLightningModule(
        model=eomt_model,
        criterion=criterion,
        num_classes=NUM_CLASSES,
        ignore_index=255,
        poly_power=0.9,
        warmup_steps=None,   # scheduler 없이 optimizer만 테스트
    )
    # Lightning log 메서드 비활성화 (Trainer 없이 호출 시 오류 방지)
    module.log = lambda *args, **kwargs: None
    module.log_dict = lambda *args, **kwargs: None
    return module


def _make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 3, H, W) 이미지 + (B, H, W) 마스크 레이블."""
    x = torch.randn(BATCH, 3, IMG_H, IMG_W)
    # 클래스 0~3, 일부는 ignore_index=255
    y = torch.randint(0, NUM_CLASSES, (BATCH, IMG_H, IMG_W))
    y[0, :4, :4] = 255  # ignore 영역 포함
    return x, y


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_eomt_forward_output_shape(eomt_model: EoMT) -> None:
    """forward()가 (mask_logits_list, class_logits_list)를 올바른 shape으로 반환하는지 확인."""
    x = torch.randn(BATCH, 3, IMG_H, IMG_W)
    with torch.no_grad():
        mask_list, class_list = eomt_model(x)

    num_l2 = len(eomt_model.backbone.l2_blocks)
    expected_outputs = num_l2 + 1  # 각 L2 블록 진입 전 + 최종

    assert len(mask_list) == expected_outputs, f"mask_list 길이 불일치: {len(mask_list)} != {expected_outputs}"
    assert len(class_list) == expected_outputs

    grid_h = IMG_H // PATCH_SIZE
    grid_w = IMG_W // PATCH_SIZE
    upscale_factor = 2 ** eomt_model.backbone.num_upscale  # ScaleBlock 횟수

    for mask_logits in mask_list:
        assert mask_logits.shape == (BATCH, NUM_Q, grid_h * upscale_factor, grid_w * upscale_factor), \
            f"mask shape 불일치: {mask_logits.shape}"

    for class_logits in class_list:
        assert class_logits.shape == (BATCH, NUM_Q, NUM_CLASSES + 1), \
            f"class shape 불일치: {class_logits.shape}"


# ---------------------------------------------------------------------------
# training_step
# ---------------------------------------------------------------------------

def test_eomt_training_step_returns_valid_loss(lightning_module: EoMTLightningModule) -> None:
    """training_step이 NaN/Inf 없는 스칼라 loss를 반환하는지 확인."""
    lightning_module.train()
    batch = _make_batch()
    loss = lightning_module.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "loss는 스칼라여야 합니다"
    assert not torch.isnan(loss), "loss가 NaN입니다"
    assert not torch.isinf(loss), "loss가 Inf입니다"
    assert loss.item() >= 0.0, "loss는 0 이상이어야 합니다"


# ---------------------------------------------------------------------------
# validation_step
# ---------------------------------------------------------------------------

def test_eomt_validation_step_runs_without_error(lightning_module: EoMTLightningModule) -> None:
    """validation_step이 오류 없이 실행되는지 확인."""
    lightning_module.eval()
    batch = _make_batch()
    with torch.no_grad():
        lightning_module.validation_step(batch, batch_idx=0)


# ---------------------------------------------------------------------------
# Mask Annealing
# ---------------------------------------------------------------------------

def test_mask_annealing_prob_decreases_over_steps(
    eomt_model: EoMT, criterion: EoMTLoss
) -> None:
    """on_train_batch_end 호출 시 annealing prob이 step에 따라 단조 감소하는지 확인."""
    module = EoMTLightningModule(
        model=eomt_model,
        criterion=criterion,
        num_classes=NUM_CLASSES,
        ignore_index=255,
        poly_power=0.9,
        attn_mask_annealing_start_steps=[0, 0],
        attn_mask_annealing_end_steps=[100, 200],
    )
    module.log = lambda *args, **kwargs: None
    module.log_dict = lambda *args, **kwargs: None

    # Trainer 없이 global_step을 mock으로 주입 (read-only property라 직접 설정 불가)
    probs_over_time = []
    for step in [0, 30, 60, 100]:
        with patch.object(type(module), "global_step", new_callable=PropertyMock, return_value=step):
            module.on_train_batch_end(None, None, 0)
        probs_over_time.append(eomt_model.attn_mask_probs.clone())

    # block 0: end_step=100이므로 step=100에서 prob=0
    assert probs_over_time[-1][0].item() == pytest.approx(0.0, abs=1e-4)
    # block 1: end_step=200이므로 step=100에서 아직 > 0
    assert probs_over_time[-1][1].item() > 0.0
    # prob은 단조 감소
    for i in range(len(probs_over_time) - 1):
        assert probs_over_time[i][0].item() >= probs_over_time[i + 1][0].item()
        assert probs_over_time[i][1].item() >= probs_over_time[i + 1][1].item()
