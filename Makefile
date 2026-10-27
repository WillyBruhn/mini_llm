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

# Run the CPU tests with AddressSanitizer and UndefinedBehaviorSanitizer.
# This finds out-of-bounds access, use-after-free, leaks and undefined behavior.
sanitize-cpu: $(BUILD)/test_runner_asan
	./$<

$(BUILD)/test_runner_asan: $(TEST_SRCS) $(CPU_SRCS) $(wildcard engine/include/*.h) engine/tests/test_util.h | $(BUILD)
	$(CC) $(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -o $@ $(TEST_SRCS) $(CPU_SRCS) $(LDFLAGS)

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

.PHONY: sanitize-cpu clang-format-check check format ruff-format-check ruff-check mypy file-length semgrep semgrep-test all test gpt2-test test-gpu sanitize-gpu pytorch-test gpu inference-lib profile-nsys profile-ncu bench bench-gpu reference weights run clean
