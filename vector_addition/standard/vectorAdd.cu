#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// CUDA kernel for vector addition
// __global__ means that this function is called from the host and runs on the device (gpu)
// void means that the function does not return a value
// const means that the argument is not being modified by the function
// the * after the `int` types indicates that the variable is a pointer to the first element of a contiguous block of integers in memory (a vector)
// __restrict indicates that the memory pointed to by the pointer is not accessed through any other pointer
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b, int *__restrict c, int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N) c[tid] = a[tid] + b[tid];
}

// Check vector add result
// this function signature looks much different from the CUDA kernel because it runs on the host and uses C++ standard library types
// again, void means the function does not return a value
// std::vector<int> denotes a dynamic array of integers. it manages memory automatically and provides functions to access its size and elements
// the & after the type indicates that the argument is a reference to the vector, which means that the function can modify the vector without making a copy
void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
    for (int i = 0; i < a.size(); i++) {
        assert(c[i] = a[i] + b[i]);
    }
}

int main() {
    // Array size of 2^16 (65536 elements)
    // This uses a left bitwise shift to multiply 2 by itself 16 times
    // constexpr means the value is constant and can be determined at compile time rather than runtime
    constexpr int N = 1 << 16;
    constexpr size_t bytes = sizeof(int) * N;

    // Vectors for holding the host-side data
    std::vector<int> a;
    a.reserve(N);
    std::vector<int> b;
    b.reserve(N);
    std::vector<int> c;
    c.reserve(N);

    // Initialize random numbers in each vector
    for (int i = 0; i < N; i++) {
        // push_back just adds the value to the end of the vector, like append in Python
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }

    // Allocate device (GPU) memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA (1024 threads per CTA)
    // CTA stands for "cooperative thread array" and is a group of threads that can communicate and synchronize with each other
    int NUM_THREADS = 1 << 10; // 2^10 through bitwise shift = 1024

    // CTAs per grid
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // Launch the kernel on the GPU
    // Kernel calls are asynchronous
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

    // Copy sum vector from device to host
    cudaMemcpy(c.data(), d_c, cudaMemcpyDeviceToHost);

    // Check result for errors
    verify_result(a, b, c);

    // Clear memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    return 0;
}