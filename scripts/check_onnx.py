#!/usr/bin/env python3
"""Static ONNX compatibility checker for Hailo Dataflow Compiler exports."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper


SUPPORTED_ONNX_OPSETS = {8, 11, 12, 13, 14, 15, 16, 17}
SEVERITIES = ("PASS", "WARN", "REVIEW", "FAIL")
AUDIT_RULES_REFERENCE = "references/onnx-audit-rules.md"

TARGET_FAMILIES = {
    "hailo8": {
        "id": "hailo8",
        "label": "Hailo8 family",
        "dfc_version": "3.33.0",
        "reference": "references/dfc-targets.md#hailo8-family",
        "hw_arch": ["hailo8", "hailo8l", "hailo8r"],
        "default_hw_arch": "hailo8",
        "notes": [
            "Use DFC 3.x for Hailo8 devices.",
            "This is the default company target for this repository.",
        ],
    },
    "hailo10_15": {
        "id": "hailo10_15",
        "label": "Hailo 10/15 family",
        "dfc_version": "5.0.0",
        "reference": "references/dfc-targets.md#hailo-1015-family",
        "hw_arch": ["hailo10h", "hailo15h", "hailo15m", "hailo15l"],
        "default_hw_arch": "hailo10h",
        "notes": [
            "Use DFC 5.x for Hailo-10 and Hailo-15 devices.",
            "Treat this as a first-class target when planning migrations.",
        ],
    },
}

REVIEW_OPS = {
    "ArgMax": "Post-processing pattern; prefer compiling logits unless on-chip logits handling is explicit.",
    "Attention": "Attention pattern; review graph-specific parser and resource behavior.",
    "Einsum": "Equation-specific support; review unless already validated on this target family.",
    "GridSample": "Coordinate transform behavior should be checked before DFC parsing.",
    "If": "Control-flow graph needs dedicated parser review.",
    "Loop": "Control-flow graph needs dedicated parser review.",
    "NonMaxSuppression": "Post-processing pattern; model-script/runtime placement should be explicit.",
    "RNN": "Sequence graph needs dedicated parser review.",
    "GRU": "Sequence graph needs dedicated parser review.",
    "LSTM": "Sequence graph needs dedicated parser review.",
    "Scan": "Control-flow graph needs dedicated parser review.",
    "TopK": "Post-processing pattern; runtime placement should be explicit.",
}


def selected_targets(target_family: str) -> list[dict[str, Any]]:
    if target_family == "all":
        return [TARGET_FAMILIES["hailo8"], TARGET_FAMILIES["hailo10_15"]]
    return [TARGET_FAMILIES[target_family]]


def add_finding(
    findings: list[dict[str, str]],
    severity: str,
    code: str,
    message: str,
    *,
    location: str = "",
) -> None:
    finding = {"severity": severity, "code": code, "message": message}
    if location:
        finding["location"] = location
    findings.append(finding)


def tensor_type_name(value_info: onnx.ValueInfoProto) -> str:
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type == 0:
        return "UNDEFINED"
    return TensorProto.DataType.Name(tensor_type.elem_type)


def shape_info(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return {
            "name": value_info.name,
            "dtype": tensor_type_name(value_info),
            "rank": None,
            "dims": [],
            "has_dynamic_dims": False,
            "has_missing_dims": True,
            "layout": "unknown",
        }

    dims: list[int | str | None] = []
    has_dynamic_dims = False
    has_missing_dims = False
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
            has_dynamic_dims = True
        else:
            dims.append(None)
            has_missing_dims = True

    layout = "unknown"
    if len(dims) == 4:
        channels_first = dims[1] == 3
        channels_last = dims[3] == 3
        if channels_first and not channels_last:
            layout = "NCHW"
        elif channels_last and not channels_first:
            layout = "NHWC"
        elif channels_first and channels_last:
            layout = "ambiguous"

    return {
        "name": value_info.name,
        "dtype": tensor_type_name(value_info),
        "rank": len(dims),
        "dims": dims,
        "has_dynamic_dims": has_dynamic_dims,
        "has_missing_dims": has_missing_dims,
        "layout": layout,
    }


def real_graph_inputs(model: onnx.ModelProto) -> list[onnx.ValueInfoProto]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return [value for value in model.graph.input if value.name not in initializer_names]


def main_opset(opsets: list[dict[str, Any]]) -> int | None:
    for opset in opsets:
        if opset["domain"] in ("", "ai.onnx"):
            return opset["version"]
    return None


def attr_text(node: onnx.NodeProto, name: str) -> str | None:
    for attr in node.attribute:
        if attr.name == name:
            value = helper.get_attribute_value(attr)
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)
    return None


def audit_onnx(
    onnx_path: str | Path,
    *,
    target_family: str = "hailo8",
    strict: bool = False,
) -> dict[str, Any]:
    path = Path(onnx_path)
    findings: list[dict[str, str]] = []
    targets = selected_targets(target_family)
    report: dict[str, Any] = {
        "model_path": str(path),
        "target_family": target_family,
        "targets": targets,
        "strict": strict,
        "valid_onnx": False,
        "opsets": [],
        "main_opset": None,
        "inputs": [],
        "outputs": [],
        "operator_counts": {},
        "rules_reference": AUDIT_RULES_REFERENCE,
        "findings": findings,
        "summary": {severity: 0 for severity in SEVERITIES},
    }

    try:
        model = onnx.load(str(path))
    except Exception as exc:  # pragma: no cover - exception text depends on onnx internals
        add_finding(findings, "FAIL", "load_onnx", f"Could not load ONNX model: {exc}")
        summarize(report)
        return report

    try:
        onnx.checker.check_model(model, full_check=True)
        report["valid_onnx"] = True
        add_finding(findings, "PASS", "onnx_checker", "ONNX checker validation passed.")
    except Exception as exc:
        add_finding(findings, "FAIL", "onnx_checker", f"ONNX checker validation failed: {exc}")

    opsets = [
        {"domain": opset.domain or "ai.onnx", "version": opset.version}
        for opset in model.opset_import
    ]
    report["opsets"] = opsets
    standard_opset = main_opset(opsets)
    report["main_opset"] = standard_opset
    if standard_opset is None:
        add_finding(findings, "WARN", "missing_main_opset", "No ai.onnx opset import was found.")
    elif standard_opset in SUPPORTED_ONNX_OPSETS:
        add_finding(
            findings,
            "PASS",
            "supported_opset",
            f"Main ONNX opset {standard_opset} is in the DFC-supported range.",
        )
    else:
        severity = "FAIL" if strict else "WARN"
        add_finding(
            findings,
            severity,
            "unsupported_opset",
            f"Main ONNX opset {standard_opset} is outside the DFC-supported set: "
            f"{sorted(SUPPORTED_ONNX_OPSETS)}.",
        )

    for opset in opsets:
        if opset["domain"] not in ("ai.onnx", ""):
            add_finding(
                findings,
                "REVIEW",
                "custom_opset_domain",
                f"Custom opset domain '{opset['domain']}' needs DFC parser review.",
            )

    inputs = [shape_info(value) for value in real_graph_inputs(model)]
    outputs = [shape_info(value) for value in model.graph.output]
    report["inputs"] = inputs
    report["outputs"] = outputs
    inspect_inputs(inputs, findings)
    inspect_outputs(outputs, findings)

    operator_counts = Counter(node.op_type for node in model.graph.node)
    report["operator_counts"] = dict(sorted(operator_counts.items()))
    inspect_operators(model, findings, targets)

    summarize(report)
    return report


def inspect_inputs(inputs: list[dict[str, Any]], findings: list[dict[str, str]]) -> None:
    if not inputs:
        add_finding(findings, "FAIL", "missing_input", "Graph has no non-initializer inputs.")
        return

    for graph_input in inputs:
        location = f"input:{graph_input['name']}"
        if graph_input["rank"] is None:
            add_finding(findings, "WARN", "missing_input_shape", "Input shape metadata is missing.", location=location)
            continue
        if graph_input["has_dynamic_dims"] or graph_input["has_missing_dims"]:
            add_finding(
                findings,
                "WARN",
                "dynamic_input_shape",
                "Input has dynamic or unknown dimensions; prefer fixed-shape ONNX for Hailo DFC.",
                location=location,
            )
        if graph_input["rank"] != 4:
            add_finding(
                findings,
                "WARN",
                "non_4d_input",
                "Expected a 4D image tensor for segmentation deployment.",
                location=location,
            )
            continue

        dims = graph_input["dims"]
        batch = dims[0]
        if isinstance(batch, int) and batch != 1:
            add_finding(
                findings,
                "WARN",
                "batch_not_one",
                "Input batch is not fixed to 1; use batch=1 unless there is a target-specific reason.",
                location=location,
            )
        elif not isinstance(batch, int):
            add_finding(
                findings,
                "WARN",
                "dynamic_batch",
                "Input batch is not a fixed integer; use batch=1 for Hailo preflight exports.",
                location=location,
            )

        layout = graph_input["layout"]
        if layout == "NCHW":
            add_finding(findings, "PASS", "input_layout_nchw", "Input layout looks like NCHW.", location=location)
        elif layout == "NHWC":
            add_finding(
                findings,
                "REVIEW",
                "input_layout_nhwc",
                "Input layout looks like NHWC; confirm DFC parser input-format and preprocessing.",
                location=location,
            )
        else:
            add_finding(
                findings,
                "REVIEW",
                "ambiguous_input_layout",
                "Could not infer NCHW/NHWC from a 4D input; confirm parser input format.",
                location=location,
            )


def inspect_outputs(outputs: list[dict[str, Any]], findings: list[dict[str, str]]) -> None:
    if not outputs:
        add_finding(findings, "FAIL", "missing_output", "Graph has no outputs.")
        return

    for graph_output in outputs:
        location = f"output:{graph_output['name']}"
        if graph_output["rank"] is None:
            add_finding(findings, "WARN", "missing_output_shape", "Output shape metadata is missing.", location=location)
            continue
        if graph_output["has_dynamic_dims"] or graph_output["has_missing_dims"]:
            add_finding(
                findings,
                "WARN",
                "dynamic_output_shape",
                "Output has dynamic or unknown dimensions; fixed logits shape is easier to validate.",
                location=location,
            )
        if graph_output["rank"] != 4:
            add_finding(
                findings,
                "REVIEW",
                "non_4d_output",
                "Segmentation exports usually produce 4D logits; confirm this output should be compiled.",
                location=location,
            )


def inspect_operators(
    model: onnx.ModelProto,
    findings: list[dict[str, str]],
    targets: list[dict[str, Any]],
) -> None:
    target_ids = {target["id"] for target in targets}
    seen_review_ops: set[str] = set()

    for node in model.graph.node:
        if node.op_type in REVIEW_OPS and node.op_type not in seen_review_ops:
            seen_review_ops.add(node.op_type)
            add_finding(
                findings,
                "REVIEW",
                "operator_review",
                f"{node.op_type}: {REVIEW_OPS[node.op_type]} See {AUDIT_RULES_REFERENCE}.",
                location=f"op:{node.op_type}",
            )

        if "hailo10_15" in target_ids and node.op_type == "Resize":
            mode = attr_text(node, "mode")
            coord_mode = attr_text(node, "coordinate_transformation_mode")
            if mode in {"linear", "bilinear"} and coord_mode in {"half_pixel", "half_pixels"}:
                add_finding(
                    findings,
                    "REVIEW",
                    "hailo15l_resize_review",
                    f"Resize uses linear half-pixel mode; review the Hailo-15L note in {AUDIT_RULES_REFERENCE}.",
                    location=f"op:{node.name or node.op_type}",
                )


def summarize(report: dict[str, Any]) -> None:
    summary = {severity: 0 for severity in SEVERITIES}
    for finding in report["findings"]:
        summary[finding["severity"]] += 1
    report["summary"] = summary


def format_shape(io_meta: dict[str, Any]) -> str:
    if io_meta["rank"] is None:
        return "unknown"
    return "[" + ", ".join(str(dim) for dim in io_meta["dims"]) + "]"


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "Hailo Compatibility Checker",
        f"Model: {report['model_path']}",
        f"Target family: {report['target_family']}",
        f"Strict: {report['strict']}",
        "",
        "Targets:",
    ]
    for target in report["targets"]:
        lines.append(
            f"- {target['label']}: DFC {target['dfc_version']}, "
            f"reference {target['reference']}, hw_arch {', '.join(target['hw_arch'])}"
        )

    lines.extend(["", "ONNX:", f"- valid_onnx: {report['valid_onnx']}", f"- main_opset: {report['main_opset']}"])
    lines.append(f"- audit_rules: {report['rules_reference']}")
    if report["opsets"]:
        formatted_opsets = ", ".join(f"{item['domain']}={item['version']}" for item in report["opsets"])
        lines.append(f"- opsets: {formatted_opsets}")

    lines.extend(["", "Inputs:"])
    for graph_input in report["inputs"]:
        lines.append(
            f"- {graph_input['name']}: {graph_input['dtype']} {format_shape(graph_input)} "
            f"layout={graph_input['layout']}"
        )

    lines.append("")
    lines.append("Outputs:")
    for graph_output in report["outputs"]:
        lines.append(
            f"- {graph_output['name']}: {graph_output['dtype']} {format_shape(graph_output)} "
            f"layout={graph_output['layout']}"
        )

    lines.append("")
    lines.append("Operators:")
    if report["operator_counts"]:
        for op_type, count in report["operator_counts"].items():
            lines.append(f"- {op_type}: {count}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Findings:")
    for finding in report["findings"]:
        location = f" ({finding['location']})" if "location" in finding else ""
        lines.append(f"[{finding['severity']}] {finding['code']}{location}: {finding['message']}")

    lines.append("")
    summary = ", ".join(f"{severity}={report['summary'][severity]}" for severity in SEVERITIES)
    lines.append(f"Summary: {summary}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static ONNX compatibility checker for Hailo DFC exports")
    parser.add_argument("onnx_path", help="Path to the ONNX model to inspect")
    parser.add_argument(
        "--target-family",
        choices=("hailo8", "hailo10_15", "all"),
        default="hailo8",
        help="Hailo target family to include in the audit report",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Report output format",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat unsupported parser-level facts as FAIL instead of WARN",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_onnx(args.onnx_path, target_family=args.target_family, strict=args.strict)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 1 if report["summary"]["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
