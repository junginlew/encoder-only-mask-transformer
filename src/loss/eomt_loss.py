from typing import List, Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)

class EoMTLoss(Mask2FormerLoss):
    """
    Hugging Face의 Mask2FormerLoss를 상속받아 EoMT에 맞게 최적화한 Loss 클래스.
    Point Sampling을 통한 메모리 최적화와 DDP 분산 학습 동기화 지원.
    """
    def __init__(
        self,
        num_classes: int,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        no_object_coefficient: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_classes
        self.eos_coef = no_object_coefficient
        
        # '배경(No Object)'에 대한 CE Loss 페널티 가중치 설정
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        # 정답 텐서 타입 캐스팅
        mask_labels = [target["masks"].to(masks_queries_logits.dtype) for target in targets]
        class_labels = [target["labels"].long() for target in targets]

        # Hungarian Matching
        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        # 매칭 결과를 바탕으로 Loss 계산
        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        return {**loss_masks, **loss_classes} # {"loss_mask": t1, "loss_dice": t2, "loss_cross_entropy": t3}

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1) # Point Sampling 기반 BCE/Dice 계산, 1로 정규화 무력화(아래에서 처리)

        # 배치 내 마스크 개수 파악
        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(num_masks, dtype=torch.float, device=masks_queries_logits.device)

        # DDP 환경일 경우 전체 GPU의 마스크 개수를 합산하여 동기화
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1) #gpu당 평균 마스크 개수, 0 방지 위해 최소 1로 클램프

        # 전체 마스크 개수로 Loss 정규화
        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks
