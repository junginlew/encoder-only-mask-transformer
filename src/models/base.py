from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MeanMetric, MetricCollection

from aidall_seg.lr_scheduler import TwoStageWarmupPolySchedule
from aidall_seg.metrics import MeanIoU, PixelAccuracy, PixelF1
from aidall_seg.models.utils import (
    create_param_groups_with_differential_lr_and_weight_decay,
    split_params_for_weight_decay,
)


class SegmentationLightningModule(L.LightningModule):
    """
    Segmentation 모델의 기본 클래스
    분류 모델과 다르게, 모든 업데이트 단위를 에포크가 아닌 배치 단위(스텝)로 설정해야 한다.

    Args:
        criterion: 손실 함수
        num_classes: 클래스 수
    """

    decoder_attribute_names: Tuple[str, ...] = ()
    decoder_exclude_submodules: Tuple[str, ...] = ("classifier",)

    def __init__(
        self,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        num_classes: int = 1000,
        ignore_index: int = 255,
        optimizer: Optional[partial[Optimizer]] = None,
        lr_scheduler: Optional[partial[LRScheduler]] = None,
    ):
        super(SegmentationLightningModule, self).__init__()

        # PyTorch Lightning 체크포인트가 인스턴스의 인자를 저장하도록 설정
        # 또한 self.hparams로 인자에 접근할 수 있음
        self.save_hyperparameters(logger=False, ignore=["criterion", "weights_path"])

        self.criterion = criterion
        self.ignore_index = ignore_index
        self._invalid_target_warning_issued = False

        # 배치 평균 값에 대한 손실 메트릭스
        self.train_loss = MeanMetric().eval()
        self.val_loss = MeanMetric().eval()

        # Segmentation 메트릭스
        self.train_metrics = MetricCollection(
            {
                "mean_iou": MeanIoU(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
            },
            prefix="train/",
        ).eval()
        self.val_metrics = MetricCollection(
            {
                "mean_iou": MeanIoU(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                "pixel_acc": PixelAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                "pixel_f1": PixelF1(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
            },
            prefix="val/",
        ).eval()
        self.test_metrics = MetricCollection(
            {
                "mean_iou": MeanIoU(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                "pixel_acc": PixelAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                "pixel_f1": PixelF1(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
            },
            prefix="test/",
        ).eval()

    def on_train_start(self):
        """
        학습 시작 시 Validation 메트릭스 초기화
        """
        self.val_loss.reset()
        self.val_metrics.reset()

    def _sanitize_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Replace target ids outside the valid range with the ignore index."""
        if y.dtype != torch.long:
            y = y.long()

        num_classes = int(self.hparams.num_classes)
        ignore_index = getattr(self, "ignore_index", None)

        invalid = (y < 0) | (y >= num_classes)
        if ignore_index is not None:
            invalid = invalid & (y != ignore_index)
        if invalid.any():
            if ignore_index is None or ignore_index < 0:
                invalid_ids = torch.unique(y[invalid]).detach().cpu().tolist()
                raise RuntimeError(
                    "Found label ids outside the valid range with no ignore_index set: "
                    f"{invalid_ids}"
                )

            if not self._invalid_target_warning_issued:
                invalid_ids = torch.unique(y[invalid]).detach().cpu().tolist()
                rank_zero_warn(
                    "Detected invalid target ids outside the configured class range "
                    f"{invalid_ids}; remapping them to ignore_index={ignore_index}."
                )
                self._invalid_target_warning_issued = True

            y = y.clone()
            y[invalid] = ignore_index

        return y

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        학습 단계에서 손실 계산 및 메트릭스 업데이트

        Args:
            batch: 배치 데이터 (x, y)
            batch_idx: 배치 인덱스

        Returns:
            loss: 손실 값 (스칼라 텐서)
        """
        x, y = batch
        y = self._sanitize_targets(y)
        outputs = self(x)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        loss = self.criterion(outputs, y)

        self.train_loss.update(loss)
        self.train_metrics.update(logits, y)

        self.log(
            "train/loss",
            self.train_loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            self.train_metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        검증 손실 계산 및 메트릭스 업데이트

        Args:
            batch: 배치 데이터 (x, y)
            batch_idx: 배치 인덱스
        """
        x, y = batch
        y = self._sanitize_targets(y)
        outputs = self(x)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        loss = self.criterion(outputs, y)

        self.val_loss.update(loss)
        self.val_metrics.update(logits, y)

        self.log(
            "val/loss",
            self.val_loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            self.val_metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        테스트 단계, 메트릭스 업데이트

        Args:
            batch: 배치 데이터 (x, y)
            batch_idx: 배치 인덱스
        """
        x, y = batch
        y = self._sanitize_targets(y)
        outputs = self(x)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

        self.test_metrics.update(logits, y)

        self.log_dict(
            self.test_metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        예측 단계이다, PyTorch Lightning의 Trainer를 이용해서 예측할때만 사용되고, 일반적인 Inference 단계에서는 사용되지 않는다.

        Args:
            batch: 배치 데이터 (x, y)
            batch_idx: 배치 인덱스

        Returns:
            x: 입력 데이터
            y: 실제 레이블
            preds: 예측 결과
        """
        x, y = batch
        # B, num_classes, H, W 형태의 로짓 텐서
        logits = self(x)
        # Softmax 적용 없이 채널 방향으로 Argmax 적용, B, H, W 형태의 라벨 인덱스 텐서
        preds = torch.argmax(logits, dim=1)

        return x, y, preds

    # ------------------------------------------------------------------
    # Decoder weight utilities
    # ------------------------------------------------------------------

    def _resolve_module_by_path(self, attr_path: str) -> nn.Module:
        """Resolve a dotted attribute path to a module instance."""
        module: Any = self
        for attr in attr_path.split('.'):
            if not hasattr(module, attr):
                raise AttributeError(
                    f"Attribute '{attr}' not found while resolving '{attr_path}'."
                )
            module = getattr(module, attr)

        if not isinstance(module, nn.Module):
            raise TypeError(
                f"Resolved object for '{attr_path}' is not an nn.Module: {type(module)}"
            )

        return module

    def _resolve_decoder_paths(
        self, attr_paths: Optional[Sequence[str]] = None
    ) -> Tuple[str, ...]:
        paths = tuple(attr_paths) if attr_paths is not None else self.decoder_attribute_names
        if not paths:
            raise ValueError(
                "Decoder attribute names are not defined. "
                "Provide attr_paths or set decoder_attribute_names on the model."
            )
        return paths

    def get_decoder_state_dict(
        self,
        attr_paths: Optional[Sequence[str]] = None,
        exclude_submodules: Optional[Sequence[str]] = None,
    ) -> "OrderedDict[str, torch.Tensor]":
        """Collect decoder weights, optionally excluding classifier-like modules."""

        resolved_paths = self._resolve_decoder_paths(attr_paths)
        excludes = (
            tuple(exclude_submodules)
            if exclude_submodules is not None
            else self.decoder_exclude_submodules
        )

        state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        for attr_path in resolved_paths:
            module = self._resolve_module_by_path(attr_path)
            module_state = module.state_dict()
            for key, value in module_state.items():
                submodule_name = key.split('.', 1)[0]
                if submodule_name and submodule_name in excludes:
                    continue
                full_key = f"{attr_path}.{key}" if attr_path else key
                state_dict[full_key] = value.detach().cpu()

        return state_dict

    def save_decoder_weights(
        self,
        output_path: Union[str, Path],
        attr_paths: Optional[Sequence[str]] = None,
        exclude_submodules: Optional[Sequence[str]] = None,
    ) -> None:
        """Save decoder weights to disk."""

        state_dict = self.get_decoder_state_dict(
            attr_paths=attr_paths, exclude_submodules=exclude_submodules
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, output_path)

    def load_decoder_weights(
        self,
        source: Union[str, Path, Mapping[str, torch.Tensor]],
        attr_paths: Optional[Sequence[str]] = None,
        strict: bool = False,
        map_location: Union[str, torch.device] = "cpu",
        exclude_submodules: Optional[Sequence[str]] = None,
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Load decoder weights from a file or state dict.

        Returns:
            Tuple containing missing and unexpected keys with full decoder prefixes.
        """

        if isinstance(source, (str, Path)):
            state_dict = torch.load(source, map_location=map_location)
        else:
            state_dict = source

        resolved_paths = self._resolve_decoder_paths(attr_paths)
        excludes = (
            tuple(exclude_submodules)
            if exclude_submodules is not None
            else self.decoder_exclude_submodules
        )

        missing: list[str] = []
        unexpected: list[str] = []

        for attr_path in resolved_paths:
            module = self._resolve_module_by_path(attr_path)
            prefix = f"{attr_path}." if attr_path else ""
            sub_state = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if not sub_state:
                continue
            m, u = module.load_state_dict(sub_state, strict=strict)

            filtered_missing = []
            for key in m:
                submodule_name = key.split('.', 1)[0]
                if submodule_name in excludes:
                    continue
                filtered_missing.append(key)

            missing.extend(f"{attr_path}.{k}" for k in filtered_missing)
            unexpected.extend(f"{attr_path}.{k}" for k in u)

        return tuple(missing), tuple(unexpected)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        최적화 및 학습률 스케줄러 설정
        Differential learning rate와 기존 기능을 모두 지원합니다.

        Returns:
            Dict[str, Any]: 최적화 및 학습률 스케줄러 설정
        """

        if self.hparams.optimizer:
            optimizer_partial = self.hparams.optimizer
            optimizer_keywords = optimizer_partial.keywords

            # Differential learning rate 체크 (multiplier 기반)
            has_backbone_multiplier = "backbone_lr_multiplier" in optimizer_keywords
            classifier_lr_value = optimizer_keywords.get("classifier_lr_multiplier")
            classifier_keywords = optimizer_keywords.get("classifier_keywords")
            has_classifier_multiplier = classifier_lr_value is not None
            weight_decay = optimizer_keywords.get("weight_decay", 0.0)
            use_differential_lr = has_backbone_multiplier or has_classifier_multiplier

            if use_differential_lr:
                # Differential learning rate 적용 (multiplier 기반)
                base_lr = optimizer_keywords["lr"]
                backbone_lr_multiplier = optimizer_keywords.get(
                    "backbone_lr_multiplier", 1.0
                )
                classifier_lr_multiplier = (
                    classifier_lr_value if has_classifier_multiplier else None
                )

                # Differential LR + Weight Decay 동시 적용
                param_groups = (
                    create_param_groups_with_differential_lr_and_weight_decay(
                        model=self.trainer.model,
                        base_lr=base_lr,
                        backbone_lr_multiplier=backbone_lr_multiplier,
                        weight_decay=weight_decay if weight_decay is not None else 0.0,
                        classifier_lr_multiplier=classifier_lr_multiplier,
                        classifier_keywords=classifier_keywords,
                    )
                )

                # optimizer 생성 시 multiplier 설정은 제외 (파라미터 그룹에서 처리됨)
                optimizer_kwargs = {
                    k: v
                    for k, v in optimizer_keywords.items()
                    if k
                    not in [
                        "backbone_lr_multiplier",
                        "classifier_lr_multiplier",
                        "classifier_keywords",
                        "lr",
                        "weight_decay",
                    ]
                }

                optimizer = optimizer_partial.func(param_groups, **optimizer_kwargs)

                # 로깅을 위한 정보 출력
                backbone_lr = base_lr * backbone_lr_multiplier
                print(f" Differential Learning Rate Applied:")
                print(f"   Base LR (Head): {base_lr}")
                if has_backbone_multiplier or backbone_lr_multiplier != 1.0:
                    print(f"   Backbone Multiplier: {backbone_lr_multiplier}")
                    print(f"   Backbone LR: {backbone_lr}")
                if classifier_lr_multiplier is not None:
                    classifier_lr = base_lr * classifier_lr_multiplier
                    print(f"   Classifier Multiplier: {classifier_lr_multiplier}")
                    print(f"   Classifier LR: {classifier_lr}")
                print(
                    f"   Weight Decay: {weight_decay if weight_decay is not None else 0.0}"
                )
                print(f"   Parameter Groups: {len(param_groups)}")

            else:
                # 기존 방식: Weight decay만 적용
                if weight_decay is not None and weight_decay > 0.0:
                    params_with_decay, params_without_decay = (
                        split_params_for_weight_decay(self.trainer.model)
                    )
                    optimizer = optimizer_partial(
                        params=[
                            {"params": params_without_decay, "weight_decay": 0.0},
                            {"params": params_with_decay, "weight_decay": weight_decay},
                        ]
                    )
                else:
                    optimizer = optimizer_partial(
                        params=self.trainer.model.parameters()
                    )
        else:
            # 기본 옵티마이저
            optimizer = torch.optim.Adam(params=self.trainer.model.parameters())

        # 학습률 스케줄러 설정 (기존과 동일)
        if self.hparams.lr_scheduler:
            scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            # LR Scheduler가 스텝 단위로 업데이트되도록 설정한다.
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

class EoMTLightningModule(SegmentationLightningModule):
    """
    Mask-Classification 기반의 EoMT 모델을 위한 전용 LightningModule.
    Mask Annealing, LLRD, 가중치 적용 복합 손실 포함.
    """
    
    decoder_attribute_names = ("model.q", "model.class_head", "model.mask_head", "model.upscale")
    
    def __init__(self, model: nn.Module, poly_power: float = 0.9,
                 warmup_steps: list[int] = None,
                 attn_mask_annealing_start_steps=None,
                 attn_mask_annealing_end_steps=None,
                 *args, **kwargs): # poly_power: mask annealing, LR poly decay의 감소 곡선 조절
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, ignore=["model", "criterion"])
        self.model = model

    def forward(self, x):
        return self.model(x)

    def _prepare_targets(self, y: torch.Tensor):
        targets = []
        for i in range(y.shape[0]):
            unique_classes = torch.unique(y[i])
            unique_classes = unique_classes[unique_classes != self.ignore_index]

            masks = []
            labels = []
            for cls in unique_classes:
                masks.append(y[i] == cls)
                labels.append(cls)

            if len(labels) > 0:
                targets.append({
                    "masks": torch.stack(masks),
                    "labels": torch.stack(labels)
                })
            else:
                targets.append({
                    "masks": torch.empty((0, y.shape[1], y.shape[2]), device=y.device, dtype=torch.bool),
                    "labels": torch.empty((0,), device=y.device, dtype=torch.long)
                })
        return targets

    def _fuse_to_semantic_logits(self, mask_logits: torch.Tensor, class_logits: torch.Tensor):
        mask_probs = mask_logits.sigmoid()
        class_probs = class_logits.softmax(dim=-1)[..., :-1]
        semantic_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        return semantic_logits

    def _calculate_total_loss(self, losses: dict) -> torch.Tensor:
        total_loss = 0.0
        for k, v in losses.items():
            if "cross_entropy" in k:
                total_loss += v * self.criterion.class_coefficient
            elif "mask" in k:
                total_loss += v * self.criterion.mask_coefficient
            elif "dice" in k:
                total_loss += v * self.criterion.dice_coefficient
        return total_loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y = self._sanitize_targets(y)
        targets = self._prepare_targets(y)
        
        mask_logits_list, class_logits_list = self(x) #각 L2블록마다 출력되는 마스크와 클래스 로짓 리스트
        
        losses = self.criterion(
            masks_queries_logits=mask_logits_list[-1],
            targets=targets,
            class_queries_logits=class_logits_list[-1]
        )
        
        for i, (mask_l, class_l) in enumerate(zip(mask_logits_list[:-1], class_logits_list[:-1])):
            aux_losses = self.criterion(
                masks_queries_logits=mask_l,
                targets=targets,
                class_queries_logits=class_l
            )
            for k, v in aux_losses.items():
                losses[f"{k}_aux_{i}"] = v

        for k, v in losses.items():
            self.log(f"losses/train_{k}", v, on_step=False, on_epoch=True, sync_dist=True)

        total_loss = self._calculate_total_loss(losses)

        semantic_logits = self._fuse_to_semantic_logits(mask_logits_list[-1], class_logits_list[-1])
       
        # semantic_logits를 y의 크기에 맞게 보간
        if semantic_logits.shape[-2:] != y.shape[-2:]:
            semantic_logits = F.interpolate(semantic_logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        self.train_loss.update(total_loss)
        self.train_metrics.update(semantic_logits, y)

        self.log("train/loss_total", self.train_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(self.train_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y = self._sanitize_targets(y)
        targets = self._prepare_targets(y)
        
        mask_logits_list, class_logits_list = self(x)
        
        losses = self.criterion(
            masks_queries_logits=mask_logits_list[-1],
            targets=targets,
            class_queries_logits=class_logits_list[-1]
        )
        
        for k, v in losses.items():
            self.log(f"losses/val_{k}", v, on_step=False, on_epoch=True, sync_dist=True)
            
        total_loss = self._calculate_total_loss(losses)

        semantic_logits = self._fuse_to_semantic_logits(mask_logits_list[-1], class_logits_list[-1])
        if semantic_logits.shape[-2:] != y.shape[-2:]:
            semantic_logits = F.interpolate(semantic_logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        self.val_loss.update(total_loss)
        self.val_metrics.update(semantic_logits, y)

        self.log("val/loss_total", self.val_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(self.val_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y = self._sanitize_targets(y)

        mask_logits_list, class_logits_list = self(x)

        semantic_logits = self._fuse_to_semantic_logits(mask_logits_list[-1], class_logits_list[-1])
        if semantic_logits.shape[-2:] != y.shape[-2:]:
            semantic_logits = F.interpolate(semantic_logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        self.test_metrics.update(semantic_logits, y)
        self.log_dict(self.test_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        if not getattr(model_ref, "masked_attn_enabled", True):
            return

        if not hasattr(model_ref, "attn_mask_probs"):
            return

        current_step = self.global_step
        poly_power = self.hparams.get("poly_power", 0.9)

        start_steps = self.hparams.get("attn_mask_annealing_start_steps", None)
        end_steps = self.hparams.get("attn_mask_annealing_end_steps", None)

        num_blocks = len(model_ref.attn_mask_probs)

        if start_steps is not None and end_steps is not None:
            # 블록별 개별 start/end step으로 annealing (앞 블록이 먼저 0에 도달)
            for i in range(num_blocks):
                start = start_steps[i]
                end = end_steps[i]
                if current_step < start:
                    prob = 1.0
                elif current_step >= end:
                    prob = 0.0
                else:
                    progress = (current_step - start) / (end - start)
                    prob = max(0.0, (1.0 - progress) ** poly_power)
                model_ref.attn_mask_probs[i] = prob
        else:
            # fallback: 모든 블록 동시에 uniform decay
            total_steps = self.trainer.estimated_stepping_batches
            progress = current_step / max(1, total_steps)
            decay_prob = max(0.0, (1.0 - progress) ** poly_power)
            model_ref.attn_mask_probs.fill_(decay_prob)

    def configure_optimizers(self):
        if not self.hparams.optimizer:
            return super().configure_optimizers()

        optimizer_partial = self.hparams.optimizer
        optimizer_keywords = optimizer_partial.keywords
        base_lr = optimizer_keywords.get("lr", 1e-4)
        weight_decay = optimizer_keywords.get("weight_decay", 0.0)
        llrd_decay = optimizer_keywords.get("llrd_decay", 0.9)
        llrd_l2_exempt = optimizer_keywords.get("llrd_l2_exempt", False)

        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        param_groups = []
        decoder_params = []

        # Decoder 파라미터 분리 (backbone이 아닌 모든 파라미터)
        for name, param in model_ref.named_parameters():
            if not param.requires_grad:
                continue
            if not name.startswith("backbone."):
                decoder_params.append(param)

        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": base_lr, "weight_decay": weight_decay})

        num_non_backbone_groups = len(param_groups)  # = 1 (decoder_params)

        # Backbone 파라미터 처리
        if hasattr(model_ref, "backbone") and hasattr(model_ref.backbone, "backbone"):
            blocks = model_ref.backbone.backbone.blocks
            num_blocks = len(blocks)
            l2_start = getattr(model_ref.backbone, "l2_start", num_blocks)

            # Transformer 블록별 LLRD 적용
            # L2 블록(l2_start 이후): llrd_l2_exempt=True면 base_lr 복원
            for i, block in enumerate(blocks):
                if llrd_l2_exempt and i >= l2_start:
                    layer_lr = base_lr
                else:
                    layer_lr = base_lr * (llrd_decay ** (num_blocks - 1 - i))
                block_params = [p for p in block.parameters() if p.requires_grad]
                if block_params:
                    param_groups.append({"params": block_params, "lr": layer_lr, "weight_decay": weight_decay})

            # 임베딩 파라미터와 최종 Norm 파라미터 분리
            embed_lr = base_lr * (llrd_decay ** num_blocks)
            embed_params = []
            norm_params = []

            for n, p in model_ref.backbone.named_parameters():
                if not p.requires_grad:
                    continue
                if "blocks." in n:
                    continue
                elif n in ["norm.weight", "norm.bias", "backbone.norm.weight", "backbone.norm.bias"]:
                    norm_params.append(p)
                else:
                    embed_params.append(p)

            if embed_params:
                param_groups.append({"params": embed_params, "lr": embed_lr, "weight_decay": weight_decay})
            if norm_params:
                param_groups.append({"params": norm_params, "lr": base_lr, "weight_decay": weight_decay})

        optimizer_kwargs = {k: v for k, v in optimizer_keywords.items() if k not in ["lr", "weight_decay", "llrd_decay", "llrd_l2_exempt"]}
        optimizer = optimizer_partial.func(param_groups, **optimizer_kwargs)

        warmup_steps = self.hparams.get("warmup_steps", None)
        if warmup_steps is not None:
            scheduler = TwoStageWarmupPolySchedule(
                optimizer=optimizer,
                num_non_backbone_groups=num_non_backbone_groups,
                warmup_steps=warmup_steps,
                total_steps=self.trainer.max_steps,
                poly_power=self.hparams.poly_power,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        if self.hparams.lr_scheduler:
            scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_total",
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
