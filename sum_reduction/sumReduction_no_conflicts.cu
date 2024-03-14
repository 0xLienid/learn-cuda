#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

const int SIZE = 256;
const int SHMEM_SIZE = 256 * 4;

__global__ void sum_reduction(int *v, int *v_r) {
    __shared__ int partial_sum[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write its result to main memory
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

void initialize_vector(int *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = 1;
    }
}

int main() {
    constexpr int N = 1 << 16;
    constexpr size_t bytes = N * sizeof(int);

    int *h_v, *h_v_r;
    int *d_v, *d_v_r;

    h_v = (int *)malloc(bytes);
    h_v_r = (int *)malloc(bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);

    initialize_vector(h_v, N);

    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    int THREAD_SIZE = SIZE;
    int GRID_SIZE = (int)ceil(N / THREAD_SIZE);

    sum_reduction<<<GRID_SIZE, THREAD_SIZE>>>(d_v, d_v_r);
    sum_reduction<<<1, THREAD_SIZE>>>(d_v_r, d_v_r);

    cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    assert(h_v_r[0] == 65536);

    free(h_v);
    free(h_v_r);
    cudaFree(d_v);
    cudaFree(d_v_r);

    printf("COMPLETED SUCCESSFULLY\n");

    return 0;
}
