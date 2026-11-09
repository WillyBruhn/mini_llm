"""Entry point of the scheduler server: `uv run scripts/serve_batched.py` or `make serve-batched`."""

import argparse
import logging
from pathlib import Path

import uvicorn
from transformers import AutoTokenizer

from mini_llm.batched_service import BatchedService, create_batched_app
from mini_llm.native import EngineOptions, NativeGPT2
from mini_llm.scheduler import Scheduler

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Serve native GPT-2 with continuous batching"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=str(root / "models/gpt2"))
    parser.add_argument(
        "--weights",
        default=str(root / "engine/tests/reference_data/weights"),
        help="Directory created by `make weights`",
    )
    parser.add_argument(
        "--library", default=None, help="Default: build/libmini_llm_gpu.so"
    )
    parser.add_argument(
        "--fp16-weights",
        action="store_true",
        help="Decode with fp16 weights (the prompt still uses fp32)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=16,
        help="Sequences decoded together in one step (1 to 16). Each holds a 72 MB KV cache",
    )
    parser.add_argument(
        "--max-prefills-per-step",
        type=int,
        default=1,
        help="New prompts read per scheduler iteration. More: lower time to first token for "
        "new requests, higher inter-token latency for the running ones",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    if not 1 <= args.max_batch <= 16:
        raise SystemExit("--max-batch must be 1 to 16")
    logger.info("Loading GPT-2 tokenizer from %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model, local_files_only=True, use_fast=True
    )
    engine = NativeGPT2(
        args.weights,
        args.library,
        "gpu",
        EngineOptions(max_sessions=args.max_batch, fp16_decode=args.fp16_weights),
    )
    scheduler = Scheduler(
        engine,
        max_batch=args.max_batch,
        max_prefills_per_step=args.max_prefills_per_step,
    )
    name = f"gpt2-small-native-gpu-batched{args.max_batch}"
    name += "-fp16" if args.fp16_weights else ""
    service = BatchedService(scheduler, tokenizer, name)
    try:
        uvicorn.run(create_batched_app(service), host=args.host, port=args.port)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
