from typing import Optional

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import DictConfig
import torch


class CallbackStateInit(Callback):
    def __init__(self, cfg: Optional[DictConfig] = None):
        self.cfg = cfg

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # 훈련 중단 플래그 초기화
        trainer.should_stop = False

        has_early_stopping = False
        patience = None
        if self.cfg is None:
            return
        # 컨피그로부터 새로운 patience 값 가져오기
        for key, config in self.cfg.items():
            if not hasattr(config, "get"):
                continue
            if (
                config.get("_target_")
                and config.get("_target_") == "lightning.pytorch.callbacks.EarlyStopping"
            ):
                has_early_stopping = True
                patience = config.get("patience", None)

        # EarlyStopping 콜백의 상태를 초기화
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                if has_early_stopping:
                    cb.wait_count = 0
                    cb.stopped_epoch = 0
                    torch_inf = torch.tensor(torch.inf)
                    cb.best_score = torch_inf if cb.monitor_op == torch.lt else -torch_inf

                    if patience:
                        cb.patience = patience
                else:
                    trainer.callbacks.remove(cb)
