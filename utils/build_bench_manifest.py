#!/usr/bin/env python3
"""Build a machine-readable manifest for NPUKernelBench cases."""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaseEntry:
    schema_version: int
    level: str
    level_num: int
    case_id: int
    op_name: str
    op_slug: str
    category: str
    category_source: str
    op_file: str
    json_file: str | None


def _slugify_op(op_name: str) -> str:
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", op_name)
    return "_".join(p.lower() for p in parts if p)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def classify_category(op_name: str) -> tuple[str, str]:
    """Classify by operation semantics, using MultiKernelBench-like buckets first."""
    name = op_name.lower()
    slug = _slugify_op(op_name)
    text = f"{name} {slug}"

    if _contains_any(text, ("moe", "mixture_of_experts")):
        return "moe", "rule:moe"
    if _contains_any(text, ("flashattn", "attention", "kv_cache", "kvcache", "rope", "rotary", "qk_norm")):
        return "attention", "rule:attention"
    if _contains_any(text, ("quant", "dequant")):
        return "quantization", "rule:quantization"
    if _contains_any(text, ("conv", "convolution")):
        return "convolution", "rule:convolution"
    if _contains_any(text, ("matmul", "batchmatmul", "bmm", "gemm")):
        return "matmul", "rule:matmul"
    if _contains_any(text, ("ffn", "fused", "fusion", "advance_step")):
        return "fuse", "rule:fuse"
    if _contains_any(text, ("layernorm", "layer_norm", "groupnorm", "group_norm", "rmsnorm", "rms_norm", "instancenorm", "instance_norm", "batchnorm", "batch_norm")):
        return "normalization", "rule:normalization"
    if _contains_any(text, ("gelu", "swiglu", "swish", "softmax", "tanh", "sigmoid", "relu", "activation")):
        return "activation", "rule:activation"
    if _contains_any(text, ("avgpool", "maxpool", "pool")):
        return "pooling", "rule:pooling"
    if _contains_any(text, ("nllloss", "loss")):
        return "loss", "rule:loss"
    if _contains_any(text, ("adam", "optimizer", "sgd", "lamb", "rmsprop", "adagrad")):
        return "optimizer", "rule:optimizer"
    if _contains_any(text, ("interpolate", "resize", "upsample", "grid_sample")):
        return "resize", "rule:resize"
    if _contains_any(text, ("iou", "nms", "box", "bbox", "ball_query", "ballquery")):
        return "geometry", "rule:geometry"
    if _contains_any(text, ("gather", "scatter", "index", "embedding", "nonzero", "topk", "top_k", "sort", "cat", "split", "pad", "repeat", "permute", "transpose", "slice", "sparse", "unfold")):
        return "index", "rule:index"
    if _contains_any(text, ("cumsum", "cumprod", "scan")):
        return "math", "rule:math"
    if _contains_any(text, ("diag", "diagonal")):
        return "math", "rule:math"
    if _contains_any(text, ("sum", "reduce", "mean", "prod", "histc")):
        return "reduce", "rule:reduce"
    if _contains_any(text, ("lstm", "rnn", "gru", "hyena", "time_decay")):
        return "sequence", "rule:sequence"
    if _contains_any(text, ("contiguous", "flatten", "reshape", "view")):
        return "shape", "rule:shape"
    if _contains_any(text, ("add", "abs", "mul", "sub", "div", "fill", "where", "broadcast")):
        return "broadcast", "rule:broadcast"
    return "unknown", "rule:unknown"


def parse_case_file(path: Path, benchmark_dir: Path) -> CaseEntry | None:
    match = re.match(r"^(\d+)_(.+)\.py$", path.name)
    if not match:
        return None

    level_match = re.match(r"^level(\d+)$", path.parent.name)
    if not level_match:
        return None

    case_id = int(match.group(1))
    op_name = match.group(2)
    level_num = int(level_match.group(1))
    json_file = path.with_suffix(".json")
    category, source = classify_category(op_name)

    return CaseEntry(
        schema_version=SCHEMA_VERSION,
        level=path.parent.name,
        level_num=level_num,
        case_id=case_id,
        op_name=op_name,
        op_slug=_slugify_op(op_name),
        category=category,
        category_source=source,
        op_file=str(path.relative_to(benchmark_dir)),
        json_file=str(json_file.relative_to(benchmark_dir)) if json_file.exists() else None,
    )


def build_manifest(benchmark_dir: Path) -> list[CaseEntry]:
    entries: list[CaseEntry] = []
    for level_dir in sorted(benchmark_dir.glob("level*"), key=lambda p: (len(p.name), p.name)):
        if not level_dir.is_dir():
            continue
        for path in sorted(level_dir.glob("*.py"), key=lambda p: (int(p.name.split("_", 1)[0]) if p.name.split("_", 1)[0].isdigit() else 10**9, p.name)):
            entry = parse_case_file(path, benchmark_dir)
            if entry is not None:
                entries.append(entry)
    return entries


def write_json(path: Path, entries: list[CaseEntry], benchmark_dir: Path) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_dir": str(benchmark_dir),
        "case_count": len(entries),
        "cases": [asdict(e) for e in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, entries: list[CaseEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(entries[0]).keys()) if entries else [
        "schema_version",
        "level",
        "level_num",
        "case_id",
        "op_name",
        "op_slug",
        "category",
        "category_source",
        "op_file",
        "json_file",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks/NPUKernelBench"))
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/NPUKernelBench/manifest.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("benchmarks/NPUKernelBench/manifest.csv"))
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    if not benchmark_dir.is_dir():
        parser.error(f"benchmark dir does not exist: {benchmark_dir}")

    entries = build_manifest(benchmark_dir)
    write_json(args.output_json, entries, benchmark_dir)
    write_csv(args.output_csv, entries)

    by_level: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for entry in entries:
        by_level[entry.level] = by_level.get(entry.level, 0) + 1
        by_category[entry.category] = by_category.get(entry.category, 0) + 1

    print(f"wrote {len(entries)} cases")
    print(f"json: {args.output_json}")
    print(f"csv: {args.output_csv}")
    print("levels:", ", ".join(f"{k}={v}" for k, v in sorted(by_level.items())))
    print("categories:", ", ".join(f"{k}={v}" for k, v in sorted(by_category.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
