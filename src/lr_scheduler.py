import math
from typing import List, Literal

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, PolynomialLR

__all__ = ["CosineAnnealingWarmupRestarts", "LinearWarmupLR", "TwoStageWarmupPolySchedule"]


class TwoStageWarmupPolySchedule(LRScheduler):
    """
    EoMT용 2단계 Warmup + Poly Decay 스케줄러.

    Non-backbone (q, class_head, mask_head, upscale):
      0 ~ warmup_steps[0]:              선형 0 → base_lr
      warmup_steps[0] ~ total:          poly decay

    Backbone (L1, L2, embed, norm):
      0 ~ warmup_steps[0]:              frozen (lr=0)
      warmup_steps[0] ~ [0]+[1]:        선형 0 → 각 블록의 target lr (L2: base_lr, L1: base_lr × llrd^n)
      warmup_steps[0]+[1] ~ total:      poly decay
    """

    def __init__(
        self,
        optimizer,
        num_non_backbone_groups: int,
        warmup_steps: list[int],  # [non_backbone_warmup, backbone_warmup]
        total_steps: int,
        poly_power: float = 0.9,
        last_epoch: int = -1,
    ):
        self.num_non_backbone_groups = num_non_backbone_groups
        self.non_backbone_warmup = warmup_steps[0]
        self.backbone_warmup = warmup_steps[1]
        self.total_steps = total_steps
        self.poly_power = poly_power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        non_bb_w = self.non_backbone_warmup
        bb_w = self.backbone_warmup

        for i, base_lr in enumerate(self.base_lrs):
            if i < self.num_non_backbone_groups:
                # Non-backbone: 선형 warmup → poly decay
                if step < non_bb_w:
                    lr = base_lr * (step / max(1, non_bb_w))
                else:
                    adjusted = step - non_bb_w
                    max_steps = max(1, self.total_steps - non_bb_w)
                    lr = base_lr * max(0.0, (1 - adjusted / max_steps) ** self.poly_power)
            else:
                # Backbone: frozen → 선형 warmup → poly decay
                if step < non_bb_w:
                    lr = 0.0
                elif step < non_bb_w + bb_w:
                    lr = base_lr * ((step - non_bb_w) / max(1, bb_w))
                else:
                    adjusted = step - non_bb_w - bb_w
                    max_steps = max(1, self.total_steps - non_bb_w - bb_w)
                    lr = base_lr * max(0.0, (1 - adjusted / max_steps) ** self.poly_power)
            lrs.append(lr)
        return lrs
