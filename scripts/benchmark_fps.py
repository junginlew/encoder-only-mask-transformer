"""
SegFormer vs EoMT FPS 벤치마크

"""

import sys
import torch

sys.path.insert(0, "src")

from aidall_seg.models import create_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SHAPE = (1, 3, 512, 512)  # B, C, H, W
WARMUP = 50
ITERATIONS = 100


def measure_fps(model: torch.nn.Module, input_shape: tuple, label: str) -> float:
    model.eval()
    model.to(DEVICE)
    dummy = torch.randn(*input_shape, device=DEVICE)

    def forward_and_discard(dummy):
        out = model(dummy)
        if isinstance(out, dict):
            _ = list(out.values())[0]
        elif isinstance(out, (tuple, list)):
            _ = out[0]

    # Warm-up
    with torch.no_grad():
        for _ in range(WARMUP):
            forward_and_discard(dummy)

    # 측정
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        for _ in range(ITERATIONS):
            forward_and_discard(dummy)
        end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    fps = ITERATIONS / (elapsed_ms / 1000)
    latency_ms = elapsed_ms / ITERATIONS

    print(f"[{label}]")
    print(f"  FPS       : {fps:.2f}")
    print(f"  Latency   : {latency_ms:.2f} ms/frame")
    print(f"  Total     : {elapsed_ms:.1f} ms ({ITERATIONS} iters)")
    print()
    return fps


def main():
    print(f"Device     : {DEVICE}")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Warm-up    : {WARMUP} iters")
    print(f"Measure    : {ITERATIONS} iters")
    print("=" * 40)

    results = {}

    # SegFormer B2
    segformer = create_model("segformer_b2", num_classes=18)
    results["SegFormer-B2"] = measure_fps(segformer, INPUT_SHAPE, "SegFormer-B2")
    del segformer
    torch.cuda.empty_cache()

    from aidall_seg.models.eomt import EoMT
    from aidall_seg.models.backbones.plain_vit import PlainViTBackbone

    # EoMT (ViT-L)
    backbone_l = PlainViTBackbone(
        backbone_name="vit_large_patch14_reg4_dinov2",
        img_size=512,
        patch_size=16,
        pretrained=False,
    )
    eomt_l = EoMT(num_classes=18, num_q=100, backbone=backbone_l)
    results["EoMT-ViTL"] = measure_fps(eomt_l, INPUT_SHAPE, "EoMT-ViTL")
    del eomt_l
    torch.cuda.empty_cache()

    # EoMT (ViT-B)
    backbone_b = PlainViTBackbone(
        backbone_name="vit_base_patch14_reg4_dinov2",
        img_size=512,
        patch_size=16,
        pretrained=False,
    )
    eomt_b = EoMT(num_classes=18, num_q=100, backbone=backbone_b)
    results["EoMT-ViTB"] = measure_fps(eomt_b, INPUT_SHAPE, "EoMT-ViTB")
    del eomt_b
    torch.cuda.empty_cache()

    # EoMT (ViT-S)
    backbone_s = PlainViTBackbone(
        backbone_name="vit_small_patch14_reg4_dinov2",
        img_size=512,
        patch_size=16,
        pretrained=False,
    )
    eomt_s = EoMT(num_classes=18, num_q=100, backbone=backbone_s)
    results["EoMT-ViTS"] = measure_fps(eomt_s, INPUT_SHAPE, "EoMT-ViTS")
    del eomt_s
    torch.cuda.empty_cache()

    # 비교 요약
    print("=" * 40)
    print("Summary")
    print("=" * 40)
    for name, fps in results.items():
        print(f"  {name:<20}: {fps:.2f} FPS")
    segformer_fps = results.get("SegFormer-B2")
    if segformer_fps:
        for name, fps in results.items():
            if name != "SegFormer-B2":
                print(f"\n  SegFormer-B2 / {name} = {segformer_fps / fps:.2f}x")


if __name__ == "__main__":
    main()
