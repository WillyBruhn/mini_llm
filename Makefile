CC      = gcc
NVCC    = nvcc
CFLAGS  = -Wall -Wextra -O2 -std=c11 -Iengine/include -Iengine/tests
LDFLAGS = -lm
NVFLAGS = -O2 -Iengine/include -Iengine/tests
BUILD   = build
GPU     ?= 1
CUDA_SANITIZER ?= /usr/local/cuda-12.6

# Source files
CPU_SRCS = $(wildcard engine/src/*.c)
CPU_OBJS = $(patsubst engine/src/%.c,$(BUILD)/%.o,$(CPU_SRCS))
TEST_SRCS = $(filter-out engine/tests/test_gpt2.c,$(wildcard engine/tests/test_*.c))
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

test-gpu: $(BUILD)/test_runner_gpu
	./$<

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

# --- CUDA build (Phase 3+) ---
gpu: $(BUILD)/mini_llm_gpu

$(BUILD)/mini_llm_gpu: engine/src/main.c $(wildcard engine/src/*.c) $(wildcard engine/src/*.cu) | $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LDFLAGS)

# --- Profiling ---
profile-nsys: $(BUILD)/mini_llm_gpu
	CUDA_VISIBLE_DEVICES=$(GPU) nsys profile --stats=true -o $(BUILD)/report ./$<
	uv run python ../massively_parallel_programming/nsys2trace.py $(BUILD)/report.sqlite $(BUILD)/report.trace.json

profile-ncu: $(BUILD)/mini_llm_gpu
	CUDA_VISIBLE_DEVICES=$(GPU) ncu --set full -o $(BUILD)/ncu_report ./$<

# --- Benchmark ---
bench: $(BUILD)/mini_llm
	@echo "=== CPU Inference Benchmark ==="
	./$< --bench

bench-gpu: $(BUILD)/mini_llm_gpu
	@echo "=== GPU Inference Benchmark ==="
	CUDA_VISIBLE_DEVICES=$(GPU) ./$< --bench

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

$(BUILD)/libmini_llm.so: engine/src/vector_addition.c engine/include/vector_addition.h | $(BUILD)
	$(CC) -Wall -Wextra -O2 -std=c11 -fPIC -shared -Iengine/include -o $@ engine/src/vector_addition.c

# --- Lint ---
# `make check` runs every linter and does not change files.
# `make format` fixes import order and formatting.
check: ruff-format-check ruff-check semgrep-test semgrep

format:
	uv run --group dev ruff check --select I --fix .
	uv run --group dev ruff format .

ruff-format-check:
	uv run --group dev ruff format --check .

ruff-check:
	uv run --group dev ruff check .

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

.PHONY: check format ruff-format-check ruff-check semgrep semgrep-test all test gpt2-test test-gpu sanitize-gpu pytorch-test gpu inference-lib profile-nsys profile-ncu bench bench-gpu reference weights run clean
