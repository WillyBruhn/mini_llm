CC      = gcc
NVCC    = nvcc
CFLAGS  = -Wall -Wextra -O2 -std=c11 -Iengine/include -Iengine/tests
LDFLAGS = -lm
NVFLAGS = -O2 -lineinfo -gencode arch=compute_86,code=sm_86 -gencode arch=compute_61,code=sm_61 -Iengine/include -Iengine/tests
BUILD   = build
GPU     ?= 1
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null)$(shell git diff --quiet HEAD 2>/dev/null || echo -dirty)
CUDA_SANITIZER ?= /usr/local/cuda-12.6

# Source files
CPU_SRCS = $(wildcard engine/src/*.c)
CPU_OBJS = $(patsubst engine/src/%.c,$(BUILD)/%.o,$(CPU_SRCS))
TEST_SRCS = $(filter-out engine/tests/test_gpt2.c engine/tests/test_gpt2_gpu.c,$(wildcard engine/tests/test_*.c))
CUDA_SRCS = $(wildcard engine/src/*.cu)
GPU_TEST_SRCS = $(wildcard engine/tests/test_*.cu)

# Default target
all: $(BUILD)/mini_llm

# --- CPU build ---
$(BUILD)/mini_llm: $(CPU_OBJS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/%.o: engine/src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

# --- Tests ---
test: $(BUILD)/test_runner
	./$<

gpt2-test: $(BUILD)/test_gpt2
	./$<

# Run the CPU tests with AddressSanitizer and UndefinedBehaviorSanitizer.
# This finds out-of-bounds access, use-after-free, leaks and undefined behavior.
sanitize-cpu: $(BUILD)/test_runner_asan
	./$<

$(BUILD)/test_runner_asan: $(TEST_SRCS) $(CPU_SRCS) $(wildcard engine/include/*.h) engine/tests/test_util.h | $(BUILD)
	$(CC) $(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -o $@ $(TEST_SRCS) $(CPU_SRCS) $(LDFLAGS)

test-gpu: $(BUILD)/test_runner_gpu
	CUDA_VISIBLE_DEVICES=$(GPU) ./$<

# Full GPU model against the saved PyTorch reference (run `make weights reference` first).
gpt2-gpu-test: $(BUILD)/test_gpt2_gpu
	CUDA_VISIBLE_DEVICES=$(GPU) ./$<

NVLINK_DEPS = $(CUDA_SRCS) $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) $(wildcard engine/include/*.h)

$(BUILD)/test_gpt2_gpu.o: engine/tests/test_gpt2_gpu.c engine/tests/test_util.h $(wildcard engine/include/*.h) | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/test_gpt2_gpu: $(BUILD)/test_gpt2_gpu.o $(NVLINK_DEPS) | $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $< $(CUDA_SRCS) $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) $(LDFLAGS)

sanitize-gpu: $(BUILD)/test_runner_gpu_debug
	CUDA_VISIBLE_DEVICES=$(GPU) \
	LD_LIBRARY_PATH=$(CUDA_SANITIZER)/compute-sanitizer:$(CUDA_SANITIZER)/lib64:$$LD_LIBRARY_PATH \
	$(CUDA_SANITIZER)/bin/compute-sanitizer --tool memcheck ./$<

$(BUILD)/test_runner: $(TEST_SRCS) engine/tests/test_util.h $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(TEST_SRCS) $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) $(LDFLAGS)

$(BUILD)/test_gpt2: engine/tests/test_gpt2.c engine/tests/test_util.h $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ engine/tests/test_gpt2.c $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) $(LDFLAGS)

$(BUILD)/test_runner_gpu: $(GPU_TEST_SRCS) $(CUDA_SRCS) engine/tests/test_util.h $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) | $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $(GPU_TEST_SRCS) $(CUDA_SRCS) $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) $(LDFLAGS)

$(BUILD)/test_runner_gpu_debug: $(GPU_TEST_SRCS) $(CUDA_SRCS) engine/tests/test_util.h $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) | $(BUILD)
	$(NVCC) -g -G -O0 -DTEST_NO_FORK $(NVFLAGS) -o $@ $(GPU_TEST_SRCS) $(CUDA_SRCS) $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) $(LDFLAGS)

# --- Benchmark ---
# One program for both backends. Pass more options with BENCH_ARGS, for example:
#   make bench-gpu BENCH_ARGS="--prompt-tokens 128 --decode-tokens 0"
BENCH_ARGS ?=

$(BUILD)/bench.o: engine/bench/bench.c $(wildcard engine/include/*.h) | $(BUILD)
	$(CC) $(CFLAGS) -DGIT_SHA='"$(GIT_SHA)"' -DBUILD_FLAGS='"$(NVFLAGS)"' -c -o $@ $<

$(BUILD)/bench: $(BUILD)/bench.o $(NVLINK_DEPS) | $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $< $(CUDA_SRCS) $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) $(LDFLAGS)

bench-gpu: $(BUILD)/bench
	CUDA_VISIBLE_DEVICES=$(GPU) ./$< --backend gpu $(BENCH_ARGS)

# The CPU is slow without a KV cache, so the default workload is small.
bench-cpu: $(BUILD)/bench
	./$< --backend cpu --prompt-tokens 8 --decode-tokens 4 --warmup 0 --runs 3 $(BENCH_ARGS)

# --- Profiling ---
# See .local/profiling.md. --profile limits the capture to the timed runs.
# A short run is enough: the profilers slow every kernel down.
PROFILE_ARGS ?= --backend gpu --prompt-tokens 32 --decode-tokens 0 --warmup 1 --runs 2 --no-check --profile
NSYS_FLAGS = --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true
# The ncu in /usr/bin is too old for the driver on this machine.
NCU ?= /usr/local/cuda/bin/ncu
KERNEL ?= multi_head_attention_kernel

# Timeline of the whole run. Writes build/report.nsys-rep. Open it in the Nsight Systems GUI.
profile-nsys: $(BUILD)/bench
	CUDA_VISIBLE_DEVICES=$(GPU) nsys profile $(NSYS_FLAGS) --stats=true -o $(BUILD)/report ./$< $(PROFILE_ARGS)

# Deep look at ONE kernel (the first launch that matches KERNEL).
# Example: make profile-ncu KERNEL=mult_kernel
profile-ncu: $(BUILD)/bench
	CUDA_VISIBLE_DEVICES=$(GPU) $(NCU) --set full --profile-from-start off -k regex:$(KERNEL) -c 1 --force-overwrite -o $(BUILD)/ncu_$(KERNEL) ./$< $(PROFILE_ARGS)

# --- Reference data for testing ---
reference:
	uv run scripts/dump_reference.py

weights:
	uv run scripts/dump_weights.py

inference-lib: $(BUILD)/libmini_llm_inference.so

$(BUILD)/libmini_llm_inference.so: $(CPU_SRCS) $(wildcard engine/include/*.h) | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(CPU_SRCS) $(LDFLAGS)

run: $(BUILD)/mini_llm
	./$<

pytorch-test: $(BUILD)/libmini_llm.so
	uv run --group dev python -m pytest pytorch/tests

# GPU library for the Python server: the C code and the CUDA code in one .so.
# nvcc links it, so the CUDA runtime is inside. Select the GPU with CUDA_VISIBLE_DEVICES.
PIC_OBJS = $(patsubst engine/src/%.c,$(BUILD)/pic/%.o,$(CPU_SRCS))

$(BUILD)/pic/%.o: engine/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -fPIC -c -o $@ $<

gpu-lib: $(BUILD)/libmini_llm_gpu.so

$(BUILD)/libmini_llm_gpu.so: $(PIC_OBJS) $(CUDA_SRCS) $(wildcard engine/include/*.h) | $(BUILD)
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -shared -o $@ $(CUDA_SRCS) $(PIC_OBJS) $(LDFLAGS)

# Pass more options with SERVE_ARGS, for example SERVE_ARGS="--port 8001".
SERVE_ARGS ?=

serve: inference-lib
	uv run scripts/serve.py $(SERVE_ARGS)

serve-gpu: gpu-lib
	CUDA_VISIBLE_DEVICES=$(GPU) uv run scripts/serve.py --backend gpu $(SERVE_ARGS)

$(BUILD)/libmini_llm.so: engine/src/vector_addition.c engine/include/vector_addition.h | $(BUILD)
	$(CC) -Wall -Wextra -O2 -std=c11 -fPIC -shared -Iengine/include -o $@ engine/src/vector_addition.c

# --- Lint ---
# `make check` runs every linter and does not change files.
# `make format` fixes import order and formatting.
check: ruff-format-check ruff-check mypy clang-format-check file-length semgrep-test semgrep

format:
	uv run --group dev ruff check --select I --fix .
	uv run --group dev ruff format .
	$(C_FILES) | xargs uv run --group dev clang-format -i

ruff-format-check:
	uv run --group dev ruff format --check .

ruff-check:
	uv run --group dev ruff check .

# C, CUDA and header files. The .semgrep test files are not real source code.
SOURCE_GLOBS = '*.c' '*.cu' '*.h' '*.cpp'
C_FILES = git ls-files --cached --others --exclude-standard -- $(SOURCE_GLOBS) ':!.semgrep'

# Style is set in .clang-format.
clang-format-check:
	$(C_FILES) | xargs uv run --group dev clang-format --dry-run --Werror

# Complexity limits (function complexity, arguments, branches, statements and
# nesting) are ruff rules. See [tool.ruff.lint] in pyproject.toml.
# No source file may be longer than MAX_FILE_LINES lines.
MAX_FILE_LINES ?= 400
file-length:
	@(git ls-files --cached --others --exclude-standard -- '*.py' ':!.semgrep'; $(C_FILES)) | xargs wc -l | \
		awk -v max=$(MAX_FILE_LINES) '$$2 != "total" && $$1 > max { print $$2 ": " $$1 " lines (max " max ")"; bad = 1 } END { exit bad }'

# Strict type checking (mypy --strict, set in pyproject.toml).
mypy:
	uv run --group dev mypy

# Rules live in .semgrep/rules. `semgrep-test` checks the rules themselves.
semgrep:
	uv run --group dev semgrep scan --config .semgrep/rules --metrics off --disable-version-check --error .

semgrep-test:
	uv run --group dev semgrep --test --metrics off --disable-version-check .semgrep/rules

# --- Utilities ---
$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: gpu-lib serve serve-gpu all test gpt2-test sanitize-cpu test-gpu gpt2-gpu-test sanitize-gpu bench-gpu bench-cpu profile-nsys profile-ncu pytorch-test inference-lib reference weights run clean check format ruff-format-check ruff-check mypy clang-format-check file-length semgrep semgrep-test
