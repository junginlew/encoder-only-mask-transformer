"""CLI for exporting segmentation models to ONNX."""
from __future__ import annotations

import argparse

from aidall_seg.runtime import load_model_from_checkpoint
from aidall_seg.runtime.exporters.onnx import export_to_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a segmentation model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to the Lightning checkpoint (.ckpt)")
    parser.add_argument("--output", required=True, help="Destination ONNX file path")
    parser.add_argument("--model-config", help="Optional Hydra model config for fallback instantiation")
    parser.add_argument("--model-name", help="Model registry name (e.g. segformer_b0, stdc1)")
    parser.add_argument("--model-type", help="Deprecated alias for --model-name")
    parser.add_argument("--device", default="cpu", help="Device for model loading (e.g. cpu, cuda, cuda:0)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy input batch size")
    parser.add_argument("--input-channels", type=int, default=3, help="Dummy input channel count")
    parser.add_argument("--input-height", type=int, default=480, help="Dummy input height")
    parser.add_argument("--input-width", type=int, default=640, help="Dummy input width")
    parser.add_argument("--dynamic-batch", action="store_true", help="Use dynamic batch axis")
    parser.add_argument(
        "--dynamic-spatial",
        action="store_true",
        help="Use dynamic height/width axes",
    )
    parser.add_argument("--dynamic-height", action="store_true", help="Use dynamic height axis")
    parser.add_argument("--dynamic-width", action="store_true", help="Use dynamic width axis")
    parser.add_argument("--half", action="store_true", help="Export using float16 dummy input (CUDA only)")
    parser.add_argument("--no-check", action="store_true", help="Skip onnx.checker validation")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic dummy input")
    parser.add_argument(
        "--no-masked-attn",
        action="store_true",
        help="Disable masked attention annealing before export (required for Hailo DFC compatibility)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, metadata = load_model_from_checkpoint(
        args.checkpoint,
        device=args.device,
        model_name=args.model_name,
        model_type=args.model_type,
        model_config_path=args.model_config,
    )

    if args.no_masked_attn:
        model_ref = model.model.module if hasattr(model.model, "module") else model.model
        model_ref.masked_attn_enabled = False

    device = next(model.parameters()).device
    if args.half:
        if device.type != "cuda":
            raise RuntimeError("Half precision export requires a CUDA device.")
        model = model.half()

    input_shape = (
        args.batch_size,
        args.input_channels,
        args.input_height,
        args.input_width,
    )

    export_path = export_to_onnx(
        model,
        args.output,
        input_shape=input_shape,
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
        dynamic_spatial=args.dynamic_spatial,
        dynamic_height=args.dynamic_height,
        dynamic_width=args.dynamic_width,
        half_precision=args.half,
        check_model=not args.no_check,
        seed=args.seed,
    )

    print("ONNX export complete:")
    print(f"  checkpoint: {metadata['checkpoint']}")
    print(f"  model: {metadata.get('model_name', metadata.get('model_class'))}")
    print(f"  onnx_check: {'skipped' if args.no_check else 'passed'}")
    print(f"  output: {export_path}")


if __name__ == "__main__":
    main()
