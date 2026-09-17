#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

static inline size_t ceil_div(size_t value, size_t divisor) {
    return value / divisor + (value % divisor != 0);
}

#endif
