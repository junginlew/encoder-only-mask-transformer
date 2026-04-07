import os
from typing import List, Optional

import hydra
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from aidall_seg.utils import load_callbacks, load_loggers, pylogger

# 커맨드라인 로거 인스턴스 생성
# 멀티 GPU 환경에서는 랭크 0 프로세스에서만 로그 기록
# PyTorch Lightning의 Logger와는 다름
log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    """
    Hydra 설정을 사용하여 PyTorch Lightning 모델을 학습한다.

    이 함수는 제공된 경우 랜덤 시드를 설정하고, 주어진 설정에 따라 데이터 모듈, 모델, 콜백, 로거를 초기화한 후,
    학습 과정을 시작하기 위해 Trainer를 초기화한다.

    Args:
        cfg (DictConfig): 시드, 데이터, 모델, 콜백, 로거, 그리고 트레이너 파라미터를 포함하는 Hydra 설정

    Raises:
        ValueError: 콜백 또는 로거 설정이 잘못된 경우 발생
    """

    # Tensor Core 지원 GPU에서 정확도 설정
    # highest: 최대 정확도, high: 높은 정확도, medium: 중간 정확도
    # 정확도와 속도는 trade-off 관계
    torch.set_float32_matmul_precision("high")

    # 랜덤 시드 설정
    if cfg.get("seed"):
        log.info("랜덤 시드 설정 중...")
        seed_everything(cfg.seed, workers=True)

    # Initialize data module
    log.info("데이터 모듈 로드 중...")
    datamodule: LightningDataModule = instantiate(cfg.data, _recursive_=False)

    # Initialize model
    log.info("모델 로드 중...")
    model: LightningModule = instantiate(cfg.model)

    # Set checkpoint path if provided
    ckpt_path = cfg.get("checkpoints", None)
    if ckpt_path:
        log.info(f"체크포인트 경로 발견: {os.path.basename(ckpt_path)}")

    # Load callbacks and loggers using helper functions
    log.info("콜백 로드 중...")
    callbacks: List[Callback] = load_callbacks(cfg.get("callbacks"))
    log.info("로거 로드 중...")
    loggers: List[Logger] = load_loggers(cfg.get("logger"))

    # Instantiate the Trainer and start training
    log.info("트레이너 초기화 중...")
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    log.info("모델 훈련 시작...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if cfg.get("test"):
        trainer.test(model, datamodule=datamodule)

    metrics = trainer.callback_metrics
    if "val/mean_iou" in metrics:
        score = metrics["val/mean_iou"].item()
        print(f"✅ Training Finished! Best Metric (val/mean_iou): {score}")
        return score
    else:
        log.warning("검증 손실 메트릭을 찾을 수 없습니다.")
        return None

    #optuna Hyperparameter Optimization을 위한 최적화 메트릭 반환
    # if cfg.get("optimized_metric"):
    #     log.info("최적화 메트릭 탐색 시작...")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning("최적 모델이 없습니다. 현재 모델을 사용합니다.")
    #         ckpt_path = None
    #     trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    #     metrics = trainer.callback_metrics
        
    #     if cfg.optimized_metric in metrics:
    #         return_metrics = metrics[cfg.optimized_metric]
    #         log.info(f"최적화 메트릭: {return_metrics}")
            
    #         return return_metrics.item()
    #     else:
    #         log.warning("최적화 메트릭이 없습니다.")
    #         return None


if __name__ == "__main__":
    train()
