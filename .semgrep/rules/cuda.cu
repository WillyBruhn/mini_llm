// --- cu-unchecked-cuda-call ---
void unchecked(float *d, const float *h, size_t bytes) {
    // ruleid: cu-unchecked-cuda-call
    cudaMalloc((void **)&d, bytes);
    // ruleid: cu-unchecked-cuda-call
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    // ruleid: cu-unchecked-cuda-call
    cudaDeviceSynchronize();
    // ruleid: cu-unchecked-cuda-call
    cudaFree(d);

    // ok: cu-unchecked-cuda-call
    CUDA_CHECK(cudaMalloc((void **)&d, bytes));
    // ok: cu-unchecked-cuda-call
    CUDA_CHECK(
        cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    // ok: cu-unchecked-cuda-call
    CUDA_CHECK(cudaFree(d));
}

// --- cu-unchecked-launch ---
void launches(float *d, size_t n) {
    // ruleid: cu-unchecked-launch
    my_kernel<<<1, 256>>>(d, n);

    // ruleid: cu-unchecked-launch
    my_kernel<<<1, 256>>>(
        d,
        n);

    // ok: cu-unchecked-launch
    my_kernel<<<1, 256>>>(d, n);
    CUDA_CHECK_LAUNCH();

    // ok: cu-unchecked-launch
    my_kernel<<<1, 256>>>(
        d,
        n);
    CUDA_CHECK_LAUNCH();
}
