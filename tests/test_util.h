#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

typedef void (*test_fn)(void);

typedef struct {
    const char *name;
    const char *file;
    test_fn fn;
} TestCase;

#define TEST(name)                                                          \
    static void test_##name(void);                                          \
    static const TestCase test_case_##name = {                              \
            #name, __FILE__, test_##name};                                  \
    static const TestCase *const test_case_ptr_##name                       \
        __attribute__((used, section("test_cases"))) = &test_case_##name;   \
    static void test_##name(void)

#define RUN_ALL_TESTS() test_run_all()

#define TEST_MAIN()                     \
    __attribute__((weak)) int main(void) { \
        return RUN_ALL_TESTS();         \
    }

#define ASSERT_TRUE(cond)                                                   \
    do {                                                                    \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define ASSERT_NEAR(actual, expected, tol)                                      \
    do {                                                                        \
        float a_ = (actual);                                                     \
        float e_ = (expected);                                                   \
        float diff_ = fabsf(a_ - e_);                                            \
        if (diff_ > (tol)) {                                                     \
            fprintf(stderr,                                                       \
                    "FAIL %s:%d: expected %.8f, got %.8f, diff %.8f, tol %.8f\n", \
                    __FILE__, __LINE__, e_, a_, diff_, (float)(tol));            \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static inline void assert_tensor_near(
    const float *actual,
    const float *expected,
    size_t count,
    float tolerance
) {
    float max_error = 0.0f;
    size_t max_index = 0;

    for (size_t i = 0; i < count; i++) {
        float error = fabsf(actual[i] - expected[i]);
        if (error > max_error) {
            max_error = error;
            max_index = i;
        }
    }

    if (max_error > tolerance) {
        fprintf(stderr,
                "FAIL tensor mismatch: max_error=%g at index=%zu actual=%g expected=%g tolerance=%g\n",
                max_error,
                max_index,
                actual[max_index],
                expected[max_index],
                tolerance);
        exit(1);
    }
}

static inline int test_run_all(void) {
    extern const TestCase *const __start_test_cases[];
    extern const TestCase *const __stop_test_cases[];

    const TestCase *const *start = __start_test_cases;
    const TestCase *const *stop = __stop_test_cases;
    int passed = 0;
    int failed = 0;

    for (const TestCase *const *entry = start; entry < stop; entry++) {
        const TestCase *test = *entry;
        printf("RUN  %s\n", test->name);
        fflush(stdout);

        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "FAIL %s: could not fork test process\n", test->name);
            failed++;
            continue;
        }

        if (pid == 0) {
            test->fn();
            exit(0);
        }

        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            fprintf(stderr, "FAIL %s: could not wait for test process\n", test->name);
            failed++;
            continue;
        }

        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            printf("PASS %s\n", test->name);
            passed++;
        } else if (WIFSIGNALED(status)) {
            printf("FAIL %s: signal %d\n", test->name, WTERMSIG(status));
            failed++;
        } else {
            printf("FAIL %s: exit %d\n", test->name, WEXITSTATUS(status));
            failed++;
        }
    }

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}

#endif
