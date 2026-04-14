import torch
from pathlib import Path
from aidall_seg.models import EoMTLightningModule
from aidall_seg.runtime.exporters.onnx import export_to_onnx

CKPT = "outputs/2026-04-13/08-46-59/checkpoints/eomt-vitb_miou_val/mean_iou=0.7200_epoch_epoch=394.ckpt"
OUTPUT = "eomt_vitb_hailo.onnx"

print("Loading checkpoint...")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
hparams = ckpt["hyper_parameters"]
state_dict = ckpt["state_dict"]

print(f"  num_classes: {hparams['num_classes']}")
print(f"  model type: {type(hparams['model']).__name__}")

# EoMT 모델 직접 꺼내기
eomt_model = hparams["model"]

# EoMTLightningModule 래핑
lightning_module = EoMTLightningModule(
    model=eomt_model,
    num_classes=hparams["num_classes"],
    ignore_index=hparams.get("ignore_index", 255),
)
lightning_module.load_state_dict(state_dict, strict=False)
lightning_module.eval()

# masked attention 비활성화 (Hailo DFC 호환)
eomt_model.masked_attn_enabled = False
print("  masked_attn_enabled set to False")

# ONNX export
print(f"Exporting to {OUTPUT} ...")
export_path = export_to_onnx(
    lightning_module,
    OUTPUT,
    input_shape=(1, 3, 512, 512),
    opset_version=17,
    dynamic_batch=False,
    dynamic_spatial=False,
    half_precision=False,
    check_model=True,
    seed=0,
)
print(f"Done: {export_path}")
