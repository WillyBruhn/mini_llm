"""Load test for the mini_llm server: N concurrent users, then a plot.

Each user sends requests one after another (closed loop). For every
concurrency level we report TTFT, inter-token latency, end-to-end latency,
output tokens/s and requests/s. Start the server first (`make serve-gpu`).

    uv run --group dev scripts/benchmark_throughput.py --users 1 5 10 20
"""

import argparse
import asyncio
import json
import logging
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

logger = logging.getLogger(__name__)

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village by the sea,",
    "def fibonacci(n):",
    "The three most important things about machine learning are",
    "In 1969, the first humans landed on the moon and",
    "Dear John, I am writing to tell you that",
]


@dataclass
class RequestResult:
    ttft_s: float
    e2e_s: float
    tokens: int
    itl_s: list[float] = field(default_factory=list)  # gaps between tokens


@dataclass
class LevelResult:
    users: int
    requests: int
    wall_s: float
    tokens_per_s: float
    requests_per_s: float
    ttft_p50_s: float
    ttft_p95_s: float
    itl_p50_s: float
    itl_p95_s: float
    e2e_p50_s: float
    e2e_p95_s: float
    gpu_mem_mb_max: float | None
    gpu_util_pct_mean: float | None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


async def one_request(
    client: httpx.AsyncClient, prompt: str, max_new_tokens: int
) -> RequestResult:
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "stream": True}
    started = time.perf_counter()
    arrivals: list[float] = []
    async with client.stream("POST", "/generate", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(event["error"])
            if not event.get("done"):
                arrivals.append(time.perf_counter() - started)
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:], strict=False)]
    return RequestResult(arrivals[0], arrivals[-1], len(arrivals), gaps)


async def user(
    client: httpx.AsyncClient,
    rng: random.Random,
    requests: int,
    max_new_tokens: int,
) -> list[RequestResult]:
    prompts = [rng.choice(PROMPTS) for _ in range(requests)]
    return [await one_request(client, p, max_new_tokens) for p in prompts]


def read_gpu(index: int) -> tuple[float, float] | None:
    """Memory used (MB) and utilization (%) of one GPU. None without nvidia-smi."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        memory, utilization = out.splitlines()[0].split(",")
        return float(memory), float(utilization)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


async def sample_gpu(index: int, samples: list[tuple[float, float]]) -> None:
    while True:
        reading = await asyncio.to_thread(read_gpu, index)
        if reading is not None:
            samples.append(reading)
        await asyncio.sleep(0.5)


async def run_level(args: argparse.Namespace, users: int) -> LevelResult:
    samples: list[tuple[float, float]] = []
    limits = httpx.Limits(max_connections=users + 1)
    async with httpx.AsyncClient(
        base_url=args.url, timeout=None, limits=limits
    ) as client:
        sampler = asyncio.create_task(sample_gpu(args.gpu, samples))
        started = time.perf_counter()
        per_user = await asyncio.gather(
            *(
                user(
                    client,
                    random.Random(args.seed * 1000 + i),
                    args.requests_per_user,
                    args.max_new_tokens,
                )
                for i in range(users)
            )
        )
        wall = time.perf_counter() - started
        sampler.cancel()

    results = [r for rs in per_user for r in rs]
    itl = [g for r in results for g in r.itl_s]
    ttft = [r.ttft_s for r in results]
    e2e = [r.e2e_s for r in results]
    return LevelResult(
        users=users,
        requests=len(results),
        wall_s=wall,
        tokens_per_s=sum(r.tokens for r in results) / wall,
        requests_per_s=len(results) / wall,
        ttft_p50_s=percentile(ttft, 0.5),
        ttft_p95_s=percentile(ttft, 0.95),
        itl_p50_s=percentile(itl, 0.5),
        itl_p95_s=percentile(itl, 0.95),
        e2e_p50_s=percentile(e2e, 0.5),
        e2e_p95_s=percentile(e2e, 0.95),
        gpu_mem_mb_max=max((m for m, _ in samples), default=None),
        gpu_util_pct_mean=(
            statistics.fmean(u for _, u in samples) if samples else None
        ),
    )


def print_table(levels: list[LevelResult]) -> None:
    logger.info(
        f"{'users':>5} {'req':>4} {'tok/s':>7} {'req/s':>6} {'TTFT p50':>9} "
        f"{'TTFT p95':>9} {'ITL p50':>8} {'e2e p50':>8} {'e2e p95':>8} {'GPU MB':>7}"
    )
    for r in levels:
        mem = f"{r.gpu_mem_mb_max:.0f}" if r.gpu_mem_mb_max is not None else "n/a"
        logger.info(
            f"{r.users:>5} {r.requests:>4} {r.tokens_per_s:>7.1f} "
            f"{r.requests_per_s:>6.2f} {r.ttft_p50_s:>8.3f}s {r.ttft_p95_s:>8.3f}s "
            f"{r.itl_p50_s * 1000:>6.1f}ms {r.e2e_p50_s:>7.2f}s "
            f"{r.e2e_p95_s:>7.2f}s {mem:>7}"
        )


def plot(levels: list[LevelResult], path: Path, title: str) -> None:
    x = [r.users for r in levels]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title)

    def draw(ax: Axes, name: str, ylabel: str, **series: list[float]) -> None:
        for label, ys in series.items():
            ax.plot(x, ys, marker="o", label=label)
        ax.set_title(name)
        ax.set_xlabel("concurrent users")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        if len(series) > 1:
            ax.legend()

    draw(
        axes[0][0],
        "Output throughput",
        "tokens/s",
        tokens=[r.tokens_per_s for r in levels],
    )
    draw(
        axes[0][1],
        "Request throughput",
        "requests/s",
        requests=[r.requests_per_s for r in levels],
    )
    draw(
        axes[0][2],
        "TTFT",
        "seconds",
        p50=[r.ttft_p50_s for r in levels],
        p95=[r.ttft_p95_s for r in levels],
    )
    draw(
        axes[1][0],
        "Inter-token latency",
        "ms",
        p50=[r.itl_p50_s * 1000 for r in levels],
        p95=[r.itl_p95_s * 1000 for r in levels],
    )
    draw(
        axes[1][1],
        "End-to-end latency",
        "seconds",
        p50=[r.e2e_p50_s for r in levels],
        p95=[r.e2e_p95_s for r in levels],
    )
    draw(
        axes[1][2],
        "GPU memory (max)",
        "MB",
        mem=[r.gpu_mem_mb_max or 0.0 for r in levels],
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)


async def check_server(url: str) -> str:
    async with httpx.AsyncClient(base_url=url, timeout=5) as client:
        try:
            response = await client.get("/health")
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise SystemExit(f"Cannot reach the server at {url}: {error!r}") from error
        return str(response.json().get("model", "unknown"))


async def main_async(args: argparse.Namespace) -> None:
    model = await check_server(args.url)
    logger.info(f"Server model: {model}. Warming up...")
    async with httpx.AsyncClient(base_url=args.url, timeout=None) as client:
        await one_request(client, PROMPTS[0], 2)

    levels: list[LevelResult] = []
    for users in args.users:
        logger.info(f"Running {users} concurrent user(s)...")
        levels.append(await run_level(args, users))
    print_table(levels)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"throughput_{time.strftime('%Y%m%d_%H%M%S')}"
    config = {k: v for k, v in vars(args).items()}
    (out / f"{stem}.json").write_text(
        json.dumps({"config": config, "levels": [asdict(r) for r in levels]}, indent=2)
    )
    plot(levels, out / f"{stem}.png", f"{model}, max_new_tokens={args.max_new_tokens}")
    logger.info(f"Saved {out / stem}.json and .png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--requests-per-user", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="nvidia-smi index of the GPU the server uses (make serve-gpu: 1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="build/benchmarks")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
