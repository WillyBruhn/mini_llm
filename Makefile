CC      = gcc
NVCC    = nvcc
CFLAGS  = -Wall -Wextra -O2 -std=c11 -Iengine/include -Iengine/tests
LDFLAGS = -lm
NVFLAGS = -O2 -Iengine/include -Iengine/tests
BUILD   = build
GPU     ?= 1

# Source files
CPU_SRCS = $(wildcard engine/src/*.c)
CPU_OBJS = $(patsubst engine/src/%.c,$(BUILD)/%.o,$(CPU_SRCS))
TEST_SRCS = $(wildcard engine/tests/test_*.c)
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

test-gpu: $(BUILD)/test_runner_gpu
	./$<

$(BUILD)/test_runner: $(TEST_SRCS) engine/tests/test_util.h $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(TEST_SRCS) $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) $(LDFLAGS)

$(BUILD)/test_runner_gpu: $(GPU_TEST_SRCS) $(CUDA_SRCS) engine/tests/test_util.h $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) | $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $(GPU_TEST_SRCS) $(CUDA_SRCS) $(filter-out $(BUILD)/main.o,$(CPU_OBJS)) $(LDFLAGS)

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

run: $(BUILD)/mini_llm
	./$<

# --- Utilities ---
$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all test test-gpu gpu profile-nsys profile-ncu bench bench-gpu reference run clean
