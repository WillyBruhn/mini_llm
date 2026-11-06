#include "../include/cuda_check.h"

static unsigned long launch_count = 0;

unsigned long cuda_launch_count(void) {
    return launch_count;
}

void cuda_launch_counted(void) {
    launch_count++;
}
