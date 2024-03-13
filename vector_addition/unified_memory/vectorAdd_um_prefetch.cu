#include <stdio.h>
#include <cassert>
#include <iostream>

using std::cout;

// CUDA kernel for vector addition
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b, int *__restrict c, int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N) c[tid] = a[tid] + b[tid];
}

// Check vector add results
void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
    for (int i = 0; i < a.size(); i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    // Array size of 2^16 elements
    constexpr int N = 1 << 16;
    constexpr size_t bytes = sizeof(int) * N;

    // Declare unified memory pointers
    int *a, *b, *c;

    // Allocate memory for these pointers
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Get the device ID for prefetching calls
    int id = cudaGetDevice(&id);

    // Set some hints about the data and do some prefetching
    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, id);

    // Initialize data
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    // Pre-fetch 'a' and 'b' arrays to the GPU
    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    // Define block size and grid size
    int BLOCK_SIZE 1 << 10; // 1024
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch the kernel
    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

    // Wait for all previous operations before using values
    cudaDeviceSynchronize();

    // Pre-fetch to the host (CPU)
    cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    // Verify the result
    verify_result(a, b, c);

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    cout << "COMPLETED SUCCESSFULLY\n";

    return 0;
}