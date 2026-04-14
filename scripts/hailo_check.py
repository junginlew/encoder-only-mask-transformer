"""
Hailo DFC compatibility check for eomt_vitb.onnx
Run with: source /project/jeongin/hailo_env/bin/activate && python scripts/hailo_check.py
"""
import os
from hailo_sdk_client import ClientRunner

ONNX_PATH = "eomt_vitb_hailo.onnx"
HAR_PATH = "eomt_vitb.har"
TARGET_CHIP = "hailo8"

print(f"[1/3] Loading ONNX: {ONNX_PATH}")
assert os.path.exists(ONNX_PATH), f"ONNX file not found: {ONNX_PATH}"

runner = ClientRunner(hw_arch=TARGET_CHIP)

print(f"[2/3] Parsing ONNX → HAR (unsupported ops will appear here)")
runner.translate_onnx_model(
    ONNX_PATH,
    "eomt_vitb",
    start_node_names=None,
    end_node_names=None,
    net_input_shapes={"image": [1, 3, 512, 512]},
)

print(f"[3/3] Saving HAR: {HAR_PATH}")
runner.save_har(HAR_PATH)

print(f"\nDone. HAR saved to: {HAR_PATH}")
print("No unsupported op errors = Hailo8 compatible at parse level")
