"""Time per stage and decode step from an nsys report made with `make profile-nsys-decode`.

    uv run scripts/nsys_stage_breakdown.py build/report_decode.nsys-rep

The stages are the NVTX ranges of the engine (`L03 mlp_down`, `lm_head`, ...). The time is
the GPU time of the kernels that each range launched. Layer numbers are merged.
"""

import argparse
import csv
import io
import logging
import re
import subprocess
from collections import defaultdict

logger = logging.getLogger(__name__)

STEP_RANGE = "embed"  # one per decode step


def read_ranges(report: str) -> list[dict[str, str]]:
    command = [
        "nsys",
        "stats",
        "--force-export=true",  # a copied report is newer than its cached export
        "--report",
        "nvtx_gpu_proj_sum",
        "--format",
        "csv",
        report,
    ]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    start = output.index("Range")  # skip the notices before the table
    return list(csv.DictReader(io.StringIO(output[start:])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    total_ns: dict[str, float] = defaultdict(float)
    instances: dict[str, int] = defaultdict(int)
    for row in read_ranges(args.report):
        stage = re.sub(r"^:?(L\d+ )?", "", row["Range"])
        if stage == "profiled runs":
            continue
        total_ns[stage] += float(row["Total Proj Time (ns)"])
        instances[stage] += int(row["Range Instances"])

    steps = instances[STEP_RANGE]
    step_ms = sum(total_ns.values()) / steps / 1e6
    logger.info("%d decode steps, %.2f ms of kernel time per step", steps, step_ms)
    logger.info(
        "%-12s %8s %12s %10s %7s",
        "stage",
        "per step",
        "avg per call",
        "ms / step",
        "share",
    )
    for stage in sorted(total_ns, key=lambda name: -total_ns[name]):
        per_step = total_ns[stage] / steps / 1e6
        logger.info(
            "%-12s %8.0f %9.1f us %10.3f %6.1f%%",
            stage,
            instances[stage] / steps,
            total_ns[stage] / instances[stage] / 1e3,
            per_step,
            100 * per_step / step_ms,
        )


if __name__ == "__main__":
    main()
