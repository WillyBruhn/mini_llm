"""Plot benchmark_throughput.py results of several runs in one figure.

    uv run --group dev scripts/compare_benchmarks.py -o docs/kernels/images/004.png \
        "no KV cache=build/benchmarks/baseline128" "KV cache=build/benchmarks/kv128"

Each run is a directory (the newest .json in it is used) or a .json file.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

PANELS = [
    ("Output throughput", "tokens/s", "tokens_per_s", 1.0),
    ("TTFT p50", "ms", "ttft_p50_s", 1000.0),
    ("Inter-token latency p50", "ms", "itl_p50_s", 1000.0),
    ("End-to-end latency p50", "seconds", "e2e_p50_s", 1.0),
    ("GPU memory (max)", "MB", "gpu_mem_mb_max", 1.0),
]


def load(path: str) -> list[dict[str, float]]:
    target = Path(path)
    if target.is_dir():
        target = max(target.glob("*.json"), key=lambda file: file.stat().st_mtime)
    levels: list[dict[str, float]] = json.loads(target.read_text())["levels"]
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", help='"label=path" for each run')
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    runs = {}
    for item in args.runs:
        label, _, path = item.partition("=")
        runs[label] = load(path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(args.title)
    for ax, (title, unit, key, scale) in zip(axes.flat, PANELS, strict=False):
        for label, levels in runs.items():
            users = [level["users"] for level in levels]
            values = [(level[key] or 0.0) * scale for level in levels]
            ax.plot(users, values, marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("concurrent users")
        ax.set_ylabel(unit)
        ax.set_xticks(users)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
