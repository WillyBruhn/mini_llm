#ifndef TENSOR_LOADING_H
#define TENSOR_LOADING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

float *load_f32_file(const char *path, size_t expected_count);

#ifdef __cplusplus
}
#endif

#endif
