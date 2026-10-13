#!/usr/bin/env python3
"""Convert a unitree_rl_lab ``policy.onnx`` MLP into ``policy.npz`` for numpy inference.

    pip install onnx   # only this script needs it
    python scripts/export_unitree_policy.py models/unitree_rl/g1_29dof_velocity_v0

The graph must be the rsl_rl actor export: Gemm layers with Elu between them,
input ``obs`` and output ``actions``. Layer ``i`` is stored as ``w{i}`` with
shape (out, in) and ``b{i}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def convert(onnx_path: Path, out_path: Path) -> None:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    arrays: dict[str, np.ndarray] = {}
    layer = 0
    activations = set()
    for node in graph.node:
        if node.op_type == "Gemm":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if float(attrs.get("alpha", 1.0)) != 1.0 or float(attrs.get("beta", 1.0)) != 1.0:
                raise ValueError(f"{node.name}: Gemm alpha/beta other than 1 is not supported")
            weight = inits[node.input[1]].astype(np.float32)
            if not int(attrs.get("transB", 0)):
                weight = weight.T
            arrays[f"w{layer}"] = weight
            arrays[f"b{layer}"] = inits[node.input[2]].astype(np.float32)
            layer += 1
        elif node.op_type == "Elu":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            activations.add(float(attrs.get("alpha", 1.0)))
        else:
            raise ValueError(f"Unsupported op {node.op_type} in {onnx_path}")
    if activations - {1.0}:
        raise ValueError(f"Elu alpha must be 1.0, got {sorted(activations)}")
    np.savez(out_path, **arrays)
    dims = [arrays["w0"].shape[1]] + [arrays[f"w{i}"].shape[0] for i in range(layer)]
    print(f"{onnx_path} -> {out_path}: {' -> '.join(str(d) for d in dims)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy_dir", type=Path)
    args = parser.parse_args()
    convert(args.policy_dir / "policy.onnx", args.policy_dir / "policy.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
