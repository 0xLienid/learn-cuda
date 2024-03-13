#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

__global__ void matmul(const int *a, const int *b, int *c, int N) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    c[row * N + col] = 0;
    for (int k = 0; k < N; k++) {
        c[row * N + col] += a[k * N + row] * b[k * N + col];
    }
}

void transpose(vector<int> &a, vector<int> &a_t) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a.size(); j++) {
            a_t[j * n + i] = a[i * n + j];
        }
    }
}

int main() {
    // Matrix size of 1024 x 1024
    constexpr int N = 1 << 10;
    constexpr size_t bytes = n * n * sizeof(int);

    // Host matrices
    vector<int> h_a(N * N);
    vector<int> h_a_t(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
    transpose(h_a, h_a_t);

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a_t.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per block
    int BLOCK_SIZE = 32;
    
    // Blocks in each dimenstion
    int GRID_SIZE = (int)ceil(N / BLOCK_SIZE);

    // Use dim3 objects
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    
    // Launch kernel
    matmul<<<grid, threads>>>(d_a, d_b, d_c, N);

    // Copy data from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result
    verify_result(h_a, h_b, h_c, N);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}