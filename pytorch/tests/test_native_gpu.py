"""The GPU backend must give the same logits as the CPU backend."""

import threading
from pathlib import Path

import numpy as np
import pytest

from mini_llm.native import EngineOptions, NativeGPT2

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "engine/tests/reference_data/weights"
CPU_LIBRARY = ROOT / "build/libmini_llm_inference.so"
GPU_LIBRARY = ROOT / "build/libmini_llm_gpu.so"


@pytest.mark.skipif(
    not (CPU_LIBRARY.is_file() and GPU_LIBRARY.is_file() and WEIGHTS.is_dir()),
    reason="needs `make inference-lib gpu-lib weights`",
)
def test_gpu_backend_matches_cpu_backend() -> None:
    tokens = np.array([15496, 11, 995], dtype=np.int32)  # "Hello, world"
    with NativeGPT2(WEIGHTS, CPU_LIBRARY, "cpu") as cpu:
        expected = cpu.forward_last(tokens)
    with NativeGPT2(WEIGHTS, GPU_LIBRARY, "gpu") as gpu:
        actual = gpu.forward_last(tokens)
        assert gpu.backend == "gpu"

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    assert int(np.argmax(actual)) == int(np.argmax(expected))


needs_gpu = pytest.mark.skipif(
    not (GPU_LIBRARY.is_file() and WEIGHTS.is_dir()),
    reason="needs `make gpu-lib weights`",
)


def _greedy_recompute(engine: NativeGPT2, prompt: list[int], steps: int) -> list[int]:
    tokens = list(prompt)
    for _ in range(steps):
        logits = engine.forward_last(np.asarray(tokens, dtype=np.int32))
        tokens.append(int(np.argmax(logits)))
    return tokens[len(prompt) :]


def _greedy_session(engine: NativeGPT2, prompt: list[int], steps: int) -> list[int]:
    generated: list[int] = []
    with engine.open_session() as session:
        logits = session.prefill(np.asarray(prompt, dtype=np.int32))
        for step in range(steps):
            generated.append(int(np.argmax(logits)))
            if step < steps - 1:
                logits = session.decode(generated[-1])
    return generated


@needs_gpu
def test_kv_cache_generates_the_same_tokens_as_recomputing() -> None:
    prompt = [464, 3139, 286, 4881, 318]  # "The capital of France is"
    with NativeGPT2(WEIGHTS, GPU_LIBRARY, "gpu") as engine:
        assert engine.use_kv_cache
        expected = _greedy_recompute(engine, prompt, 20)
        assert _greedy_session(engine, prompt, 20) == expected


@needs_gpu
def test_sessions_reuse_a_cache_without_leaking_the_previous_sequence() -> None:
    first = [464, 3139, 286, 4881, 318]
    second = [15496, 11, 995, 0, 50, 60, 70]
    with NativeGPT2(
        WEIGHTS, GPU_LIBRARY, "gpu", EngineOptions(max_sessions=1)
    ) as engine:
        expected_first = _greedy_recompute(engine, first, 8)
        expected_second = _greedy_recompute(engine, second, 8)
        # max_sessions=1 forces the second run to reuse the first run's cache.
        assert _greedy_session(engine, first, 8) == expected_first
        assert _greedy_session(engine, second, 8) == expected_second
        assert _greedy_session(engine, first, 8) == expected_first
        assert len(engine._all_caches) == 1


@needs_gpu
def test_concurrent_sessions_do_not_mix_up_their_caches() -> None:
    prompts = [[464, 3139, 286, 4881, 318], [15496, 11, 995], [40, 588, 257, 5916]]
    results: dict[int, list[int]] = {}
    with NativeGPT2(
        WEIGHTS, GPU_LIBRARY, "gpu", EngineOptions(max_sessions=2)
    ) as engine:
        expected = [_greedy_recompute(engine, p, 10) for p in prompts]

        def run(index: int) -> None:
            results[index] = _greedy_session(engine, prompts[index], 10)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(engine._all_caches) <= 2
    assert [results[i] for i in range(len(prompts))] == expected


@needs_gpu
def test_session_gives_its_cache_back_after_an_error_and_when_closed_twice() -> None:
    with NativeGPT2(
        WEIGHTS, GPU_LIBRARY, "gpu", EngineOptions(max_sessions=1)
    ) as engine:
        with pytest.raises(RuntimeError):
            with engine.open_session() as session:
                session.decode(5)  # no prefill yet: the cache is empty
        session.close()  # closing twice must not release a slot twice
        # The only slot is free again, so this does not block.
        assert _greedy_session(engine, [464, 3139], 2)


@needs_gpu
def test_without_kv_cache_a_session_recomputes() -> None:
    prompt = [464, 3139, 286, 4881, 318]
    with NativeGPT2(
        WEIGHTS, GPU_LIBRARY, "gpu", EngineOptions(use_kv_cache=False)
    ) as engine:
        assert not engine.use_kv_cache
        assert _greedy_session(engine, prompt, 6) == _greedy_recompute(
            engine, prompt, 6
        )


@needs_gpu
def test_fp16_decode_generates_text_close_to_fp32() -> None:
    prompt = [464, 3139, 286, 4881, 318]  # "The capital of France is"
    with NativeGPT2(WEIGHTS, GPU_LIBRARY, "gpu") as reference:
        expected_logits = reference.forward_last(np.asarray(prompt, dtype=np.int32))
        expected = _greedy_session(reference, prompt, 12)
    with NativeGPT2(
        WEIGHTS, GPU_LIBRARY, "gpu", EngineOptions(fp16_decode=True)
    ) as engine:
        assert engine.fp16_decode
        with engine.open_session() as session:
            # The prompt is read with the fp32 weights, so its logits are the same.
            logits = session.prefill(np.asarray(prompt, dtype=np.int32))
            np.testing.assert_array_equal(logits, expected_logits)
        actual = _greedy_session(engine, prompt, 12)
    # One close call may change the text. For this prompt the first tokens must agree.
    assert actual[:4] == expected[:4]


@needs_gpu
def test_fp16_decode_needs_the_gpu_and_the_cache() -> None:
    with pytest.raises(ValueError):
        NativeGPT2(
            WEIGHTS,
            GPU_LIBRARY,
            "gpu",
            EngineOptions(use_kv_cache=False, fp16_decode=True),
        )
    with pytest.raises(ValueError):
        NativeGPT2(WEIGHTS, CPU_LIBRARY, "cpu", EngineOptions(fp16_decode=True))


def _queued_session(
    engine: NativeGPT2, prompt: list[int], steps: int, lookahead: bool
) -> list[int]:
    """The tokens of start/enqueue/take, with or without a step queued ahead."""
    generated: list[int] = []
    with engine.open_session() as session:
        session.start(np.asarray(prompt, dtype=np.int32))
        for step in range(steps):
            more = step < steps - 1
            if lookahead and more:
                session.enqueue()
            generated.append(session.take())
            if not lookahead and more:
                session.enqueue()
    return generated


@needs_gpu
@pytest.mark.parametrize("lookahead", [False, True])
@pytest.mark.parametrize("fp16", [False, True])
def test_queued_session_generates_the_tokens_of_the_logits_path(
    lookahead: bool, fp16: bool
) -> None:
    prompt = [464, 3139, 286, 4881, 318]
    with NativeGPT2(
        WEIGHTS, GPU_LIBRARY, "gpu", EngineOptions(fp16_decode=fp16)
    ) as engine:
        expected = _greedy_session(engine, prompt, 20)
        assert _queued_session(engine, prompt, 20, lookahead) == expected


@needs_gpu
def test_queued_session_without_a_kv_cache_generates_the_same_tokens() -> None:
    prompt = [464, 3139, 286, 4881, 318]
    with NativeGPT2(
        WEIGHTS, GPU_LIBRARY, "gpu", EngineOptions(use_kv_cache=False)
    ) as engine:
        expected = _greedy_recompute(engine, prompt, 6)
        assert _queued_session(engine, prompt, 6, lookahead=True) == expected
        assert _queued_session(engine, prompt, 6, lookahead=False) == expected


@needs_gpu
def test_queued_sessions_run_in_threads_without_mixing_their_tokens() -> None:
    prompts = [[464, 3139, 286, 4881, 318], [15496, 11, 995], [40, 588, 257, 5916]]
    results: dict[int, list[int]] = {}
    with NativeGPT2(
        WEIGHTS, GPU_LIBRARY, "gpu", EngineOptions(max_sessions=3)
    ) as engine:
        expected = [_greedy_session(engine, p, 24) for p in prompts]

        def run(index: int) -> None:
            results[index] = _queued_session(engine, prompts[index], 24, True)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert [results[i] for i in range(3)] == expected


@needs_gpu
def test_queued_session_reports_misuse_and_a_reused_cache_starts_clean() -> None:
    prompt = [464, 3139, 286, 4881, 318]
    with NativeGPT2(
        WEIGHTS, GPU_LIBRARY, "gpu", EngineOptions(max_sessions=1)
    ) as engine:
        expected = _greedy_session(engine, prompt, 8)
        with engine.open_session() as session:
            with pytest.raises(RuntimeError):
                session.enqueue()  # nothing started
            with pytest.raises(RuntimeError):
                session.take()  # nothing queued
            with pytest.raises(ValueError):
                session.start(np.zeros(0, dtype=np.int32))
            session.start(np.asarray(prompt, dtype=np.int32))
            session.enqueue()
            session.enqueue()  # two tokens stay queued and are never read
        # The same cache again: the unread tokens of the last session must be gone.
        assert _queued_session(engine, prompt, 8, lookahead=True) == expected
