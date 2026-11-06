"""Where does the GPU idle during decode steps? Reads an nsys report of `make profile-nsys-decode`.

    uv run scripts/nsys_gap_analysis.py build/report_decode.nsys-rep

A decode step starts with the embed kernel. The report shows, per step: the busy time of the
kernels, the idle time inside the step (and how much of it is the GPU waiting for the CPU to
launch the next kernel), the idle time between two steps, and how many launches a step has.
"""

import argparse
import logging
import sqlite3
import statistics
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# A kernel that starts less than this long after its launch call returned was waiting for the CPU.
LAUNCH_LAG_NS = 4000


def load(report: str) -> sqlite3.Connection:
    sqlite_path = Path(report).with_suffix(".gap.sqlite")
    subprocess.run(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite",
            "true",
            "-o",
            str(sqlite_path),
            report,
        ],
        capture_output=True,
        check=True,
    )
    return sqlite3.connect(sqlite_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db = load(args.report)
    start, end = db.execute(
        "select start, end from NVTX_EVENTS where text='profiled runs'"
    ).fetchone()
    kernels = db.execute(
        "select k.start, k.end, s.value, k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.shortName where k.start >= ? and k.end <= ? order by k.start",
        (start, end),
    ).fetchall()
    launches = {
        row[0]: row[1:]
        for row in db.execute(
            "select correlationId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and end <= ?",
            (start - 1_000_000, end),
        )
    }
    steps: list[list[tuple[int, int, str, int]]] = []
    for kernel in kernels:
        if kernel[2].startswith("embed_tokens"):
            steps.append([])
        if steps:
            steps[-1].append(kernel)
    steps = steps[1:-1]  # the first and last step touch the edge of the capture
    count = len(steps)

    busy = sum(k[1] - k[0] for step in steps for k in step) / count
    inside = waiting_for_cpu = 0.0
    by_next: dict[str, float] = {}
    for step in steps:
        for before, after in zip(step, step[1:], strict=False):
            gap = after[0] - before[1]
            if gap <= 0:
                continue
            inside += gap
            by_next[after[2][:22]] = by_next.get(after[2][:22], 0.0) + gap
            launch = launches.get(after[3])
            if launch is not None and after[0] - launch[1] < LAUNCH_LAG_NS:
                waiting_for_cpu += gap
    between = [steps[i + 1][0][0] - steps[i][-1][1] for i in range(count - 1)]
    period = (steps[-1][0][0] - steps[0][0][0]) / (count - 1)
    durations = [k[1] - k[0] for step in steps for k in step]
    launch_costs = [
        launches[k[3]][1] - launches[k[3]][0]
        for step in steps
        for k in step
        if k[3] in launches
    ]

    logger.info("steps analysed: %d, launches per step: %d", count, len(steps[0]))
    logger.info(
        "step period (first kernel to the next first kernel): %.3f ms", period / 1e6
    )
    logger.info(
        "  kernels running:                  %.3f ms  (%.1f%% of the period)",
        busy / 1e6,
        100 * busy / period,
    )
    logger.info(
        "  idle inside the step:             %.3f ms  (%.1f%%)",
        inside / count / 1e6,
        100 * inside / count / period,
    )
    logger.info(
        "    of it, GPU waited for a launch: %.3f ms  (%.0f%% of the idle time)",
        waiting_for_cpu / count / 1e6,
        100 * waiting_for_cpu / inside,
    )
    logger.info(
        "  idle between two steps:           %.3f ms  (median)",
        statistics.median(between) / 1e6,
    )
    logger.info(
        "cudaLaunchKernel on the CPU: median %.2f us",
        statistics.median(launch_costs) / 1e3,
    )
    short = sum(1 for d in durations if d < LAUNCH_LAG_NS)
    logger.info(
        "kernels shorter than %d us: %d of %d (%.0f%%)",
        LAUNCH_LAG_NS // 1000,
        short,
        len(durations),
        100 * short / len(durations),
    )
    logger.info("idle time inside the step, by the kernel that waits (ms per step):")
    for name, idle in sorted(by_next.items(), key=lambda item: -item[1])[:6]:
        logger.info("  %-24s %.3f", name, idle / count / 1e6)


if __name__ == "__main__":
    main()
