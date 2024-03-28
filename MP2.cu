
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include <device_functions.h>

const int TILE_WIDTHS[] = { 2, 5, 10, 25 };
const int NUM_WIDTHS = sizeof(TILE_WIDTHS) / sizeof(int);

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float* A, float* B, float* O, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float temp = 0;
        for (int i = 0; i < size; i++) {
            temp += A[row * size + i] * B[i * size + col];
        }
        O[row * size + col] = temp;
    }
}

// Host function to perform matrix multiplication on CPU
void matrixMulCPU(float* M, float* N, float* P, int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = sum;
        }
    }
}

// Function to initialize a matrix with random values
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }
}

// Function to compare two matrices
bool compareMatrices(float* A, float* B, int size, float tolerance) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    const int size = 1500; // Size of the square matrices
    const float tolerance = 1e-5f; // Tolerance for comparison of CPU and GPU results

    // Set random seed
    srand(time(NULL));

    // Allocate memory for matrices on host
    float* M, * N, * P_cpu, * P_gpu;
    size_t matrix_size = size * size * sizeof(float);
    M = (float*)malloc(matrix_size);
    N = (float*)malloc(matrix_size);
    P_cpu = (float*)malloc(matrix_size);
    P_gpu = (float*)malloc(matrix_size);

    // Initialize matrices M and N with random values
    initializeMatrix(M, size * size);
    initializeMatrix(N, size * size);

    // Allocate memory for matrices on device
    float* d_M, * d_N, * d_P;
    cudaMalloc(&d_M, matrix_size);
    cudaMalloc(&d_N, matrix_size);
    cudaMalloc(&d_P, matrix_size);

    for (int i = 0; i < NUM_WIDTHS; ++i) {
        int TILE_WIDTH = TILE_WIDTHS[i];

        // Copy input matrices from host to device
        cudaMemcpy(d_M, M, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, N, matrix_size, cudaMemcpyHostToDevice);

        // Define grid and block dimensions based on TILE_WIDTH
        dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);

        // Start timer
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Launch kernel for matrix multiplication on GPU
        matrixMul << <gridSize, blockSize >> > (d_M, d_N, d_P, size);

        // Stop timer
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaMemcpy(P_gpu, d_P, matrix_size, cudaMemcpyDeviceToHost);

        // Perform matrix multiplication on CPU for comparison
        matrixMulCPU(M, N, P_cpu, size);

        // Compare CPU and GPU results
        if (compareMatrices(P_cpu, P_gpu, size * size, tolerance)) {
            std::cout << "TILE_WIDTH = " << TILE_WIDTH << ": Test PASSED. Time: " << milliseconds << " ms" << std::endl;
        }
        else {
            std::cout << "TILE_WIDTH = " << TILE_WIDTH << ": Test FAILED." << std::endl;
        }
    }
        //// Print GPU multiplication result
        //std::cout << "GPU Multiplication Result:" << std::endl;
        //for (int i = 0; i < size; ++i) {
        //    for (int j = 0; j < size; ++j) {
        //        std::cout << P_gpu[i * size + j] << " ";
        //    }
        //    std::cout << std::endl;
        //}

        //// Print CPU multiplication result
        //std::cout << "CPU Multiplication Result:" << std::endl;
        //for (int i = 0; i < size; ++i) {
        //    for (int j = 0; j < size; ++j) {
        //        std::cout << P_cpu[i * size + j] << " ";
        //    }
        //    std::cout << std::endl;
        //}

    // Free memory
    free(M);
    free(N);
    free(P_cpu);
    free(P_gpu);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
