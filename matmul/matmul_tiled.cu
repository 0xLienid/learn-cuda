#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10;

__global__ void tiled_matmul(int *a, int *b, int*c) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Statically allocated shared memory
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    // Sweep tile across matrix and accumulate
    int result = 0;
    for (int i = 0; i < N; i += blockDim.x) {
        // Load in elements for this tile
        s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

        // Wait for both tiles to be loaded before doing computation
        __syncthreads();

        // Do matrix multiplication on the small matrix
        for (int j = 0; j < blockDim.x; j++) {
            result += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
        }

        // Wait for all threads to finish using current tiles before loading in new ones
        __syncthreads();
    }

    // Write out this thread's result
    c[row * N + col] = result;
}

// Check result
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }

            assert(tmp == c[i * N + j])
        }
    }
}

int main() {
    constexpr size_t bytes = N * N * sizeof(int);

    // Host vectors
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    int THREADS = 32;

    // Blocks
    int BLOCKS = (int)ceil(N / THREADS);

    // Use dim3 structs for block and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    tiled_matmul<<<blocks, threads>>>(d_a, d_b, d_c);

    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result
    verify_result(h_a, h_b, h_c);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}