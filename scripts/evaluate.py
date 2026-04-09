"""
학습된 EoMT 체크포인트로 aidall val set 성능을 측정한다.

사용법:
    python scripts/evaluate.py \
        experiment=eomt/eomt-vitl \
        checkpoints=/path/to/best.ckpt \
        data.data_dir=/path/to/data
"""

import os
from typing import List

import hydra
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from aidall_seg.utils import load_callbacks, load_loggers, pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# aidall v1.0 클래스 정의
CLASS_NAMES = [
    "background", "sidewalk", "crosswalk", "curb-cut", "terrain",
    "bike-lane", "minor-road", "parking-area", "road", "curb",
    "manhole", "drain", "cautious-zone", "person", "micromobility",
    "motorcycle", "car-four-wheeled", "other-vehicle",
]


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def evaluate(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")

    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    ckpt_path = cfg.get("checkpoints", None)
    if not ckpt_path:
        raise ValueError(
            "Checkpoint path must be specified. "
            "Example: checkpoints=/path/to/checkpoint.ckpt"
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Checkpoint loaded: {ckpt_path}")
    log.info("Loading Data Module...")
    datamodule: LightningDataModule = instantiate(cfg.data, _recursive_=False)

    log.info("Loading Model...")
    model: LightningModule = instantiate(cfg.model)

    callbacks: List = load_callbacks(cfg.get("callbacks"))
    loggers: List = load_loggers(cfg.get("logger"))

    log.info("Initializing Trainer...")
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    log.info("Starting Evaluation...")
    results = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if results:
        metrics = results[0]
        print("\n" + "=" * 50)
        print("  Evaluation Results")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"  {key:<30} {value:.4f}")
        print("=" * 50)

        miou = metrics.get("test/mean_iou", None)
        pixel_acc = metrics.get("test/pixel_acc", None)
        pixel_f1 = metrics.get("test/pixel_f1", None)

        print(f"\n  mIoU       : {miou:.4f}" if miou is not None else "  mIoU       : -")
        print(f"  Pixel Acc  : {pixel_acc:.4f}" if pixel_acc is not None else "  Pixel Acc  : -")
        print(f"  Pixel F1   : {pixel_f1:.4f}" if pixel_f1 is not None else "  Pixel F1   : -")
        print()


if __name__ == "__main__":
    evaluate()
