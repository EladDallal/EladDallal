#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdlib>
#include <ctime>

const int TILE_WIDTHS[] = { 2, 5, 10, 25 };
const int NUM_WIDTHS = sizeof(TILE_WIDTHS) / sizeof(int);

// CUDA kernel for tiled matrix multiplication
__global__ void matrixMul(float* A, float* B, float* O, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float temp = 0;
        for (int k = 0; k < colsA; ++k) {
            // Check if within matrix bounds
            if (k < colsA && row < rowsA && col < colsB) {
                temp += A[row * colsA + k] * B[k * colsB + col];
            }
        }
        O[row * colsB + col] = temp;
    }
}

// Host function to perform matrix multiplication on CPU
void matrixMulCPU(float* A, float* B, float* O, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            O[i * colsB + j] = sum;
        }
    }
}

// Function to initialize a matrix with random values
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
        }
    }
}

// Function to compare two matrices
bool compareMatrices(float* A, float* B, int rows, int cols, float tolerance) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (fabs(A[i * cols + j] - B[i * cols + j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    const int rowsA1 = 400; // Number of rows in matrix A for experiment 1
    const int colsA1 = 450; // Number of columns in matrix A for experiment 1
    const int colsB1 = 500; // Number of columns in matrix B for experiment 1

    const int rowsA2 = 1200; // Number of rows in matrix A for experiment 2
    const int colsA2 = 1350; // Number of columns in matrix A for experiment 2
    const int colsB2 = 1150; // Number of columns in matrix B for experiment 2

    const float tolerance = 1e-5f; // Tolerance for comparison of CPU and GPU results

    // Set random seed
    srand(time(NULL));

    // Allocate memory for matrices on host for experiment 1
    size_t matrix_size_A1 = rowsA1 * colsA1 * sizeof(float);
    size_t matrix_size_B1 = colsA1 * colsB1 * sizeof(float);
    size_t matrix_size_O1 = rowsA1 * colsB1 * sizeof(float);
    float* A1 = (float*)malloc(matrix_size_A1);
    float* B1 = (float*)malloc(matrix_size_B1);
    float* P_cpu1 = (float*)malloc(matrix_size_O1);
    float* P_gpu1 = (float*)malloc(matrix_size_O1);

    // Initialize matrices A and B with random values for experiment 1
    initializeMatrix(A1, rowsA1, colsA1);
    initializeMatrix(B1, colsA1, colsB1);

    // Allocate memory for matrices on device for experiment 1
    float* d_A1, * d_B1, * d_P1;
    cudaMalloc(&d_A1, matrix_size_A1);
    cudaMalloc(&d_B1, matrix_size_B1);
    cudaMalloc(&d_P1, matrix_size_O1);

    // Allocate memory for matrices on host for experiment 2
    size_t matrix_size_A2 = rowsA2 * colsA2 * sizeof(float);
    size_t matrix_size_B2 = colsA2 * colsB2 * sizeof(float);
    size_t matrix_size_O2 = rowsA2 * colsB2 * sizeof(float);
    float* A2 = (float*)malloc(matrix_size_A2);
    float* B2 = (float*)malloc(matrix_size_B2);
    float* P_cpu2 = (float*)malloc(matrix_size_O2);
    float* P_gpu2 = (float*)malloc(matrix_size_O2);

    // Initialize matrices A and B with random values for experiment 2
    initializeMatrix(A2, rowsA2, colsA2);
    initializeMatrix(B2, colsA2, colsB2);

    // Allocate memory for matrices on device for experiment 2
    float* d_A2, * d_B2, * d_P2;
    cudaMalloc(&d_A2, matrix_size_A2);
    cudaMalloc(&d_B2, matrix_size_B2);
    cudaMalloc(&d_P2, matrix_size_O2);

    for (int i = 0; i < NUM_WIDTHS; ++i) {
        int TILE_WIDTH = TILE_WIDTHS[i];

        // Experiment 1
        // Copy input matrices from host to device for experiment 1
        cudaMemcpy(d_A1, A1, matrix_size_A1, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B1, B1, matrix_size_B1, cudaMemcpyHostToDevice);

        // Define grid and block dimensions based on TILE_WIDTH for experiment 1
        dim3 blockSize1(TILE_WIDTH, TILE_WIDTH);
        dim3 gridSize1((colsB1 + blockSize1.x - 1) / blockSize1.x, (rowsA1 + blockSize1.y - 1) / blockSize1.y);

        // Start timer for experiment 1
        cudaEvent_t start1, stop1;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
        cudaEventRecord(start1);

        // Launch kernel for tiled matrix multiplication on GPU for experiment 1
        matrixMul << <gridSize1, blockSize1 >> > (d_A1, d_B1, d_P1, rowsA1, colsA1, colsB1);

        // Stop timer for experiment 1
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        float milliseconds1 = 0;
        cudaEventElapsedTime(&milliseconds1, start1, stop1);

        // Copy output matrix from device to host for experiment 1
        cudaMemcpy(P_gpu1, d_P1, matrix_size_O1, cudaMemcpyDeviceToHost);

        // Perform matrix multiplication on CPU for comparison for experiment 1
        matrixMulCPU(A1, B1, P_cpu1, rowsA1, colsA1, colsB1);

        // Compare CPU and GPU results for experiment 1
        if (compareMatrices(P_cpu1, P_gpu1, rowsA1, colsB1, tolerance)) {
            std::cout << "Experiment 1 - TILE_WIDTH = " << TILE_WIDTH << ": Test PASSED. Time: " << milliseconds1 << " ms" << std::endl;
        }
        else {
            std::cout << "Experiment 1 - TILE_WIDTH = " << TILE_WIDTH << ": Test FAILED." << std::endl;
        }

        // Experiment 2
        // Copy input matrices from host to device for experiment 2
        cudaMemcpy(d_A2, A2, matrix_size_A2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B2, B2, matrix_size_B2, cudaMemcpyHostToDevice);

        // Define grid and block dimensions based on TILE_WIDTH for experiment 2
        dim3 blockSize2(TILE_WIDTH, TILE_WIDTH);
        dim3 gridSize2((colsB2 + blockSize2.x - 1) / blockSize2.x, (rowsA2 + blockSize2.y - 1) / blockSize2.y);

        // Start timer for experiment 2
        cudaEvent_t start2, stop2;
        cudaEventCreate(&start2);
        cudaEventCreate(&stop2);
        cudaEventRecord(start2);

        // Launch kernel for tiled matrix multiplication on GPU for experiment 2
        matrixMul << <gridSize2, blockSize2 >> > (d_A2, d_B2, d_P2, rowsA2, colsA2, colsB2);

        // Stop timer for experiment 2
        cudaEventRecord(stop2);
        cudaEventSynchronize(stop2);
        float milliseconds2 = 0;
        cudaEventElapsedTime(&milliseconds2, start2, stop2);

        // Copy output matrix from device to host for experiment 2
        cudaMemcpy(P_gpu2, d_P2, matrix_size_O2, cudaMemcpyDeviceToHost);

        // Perform matrix multiplication on CPU for comparison for experiment 2
        matrixMulCPU(A2, B2, P_cpu2, rowsA2, colsA2, colsB2);

        // Compare CPU and GPU results for experiment 2
        if (compareMatrices(P_cpu2, P_gpu2, rowsA2, colsB2, tolerance)) {
            std::cout << "Experiment 2 - TILE_WIDTH = " << TILE_WIDTH << ": Test PASSED. Time: " << milliseconds2 << " ms" << std::endl;
        }
        else {
            std::cout << "Experiment 2 - TILE_WIDTH = " << TILE_WIDTH << ": Test FAILED." << std::endl;
        }
    }

    // Free memory for experiment 1
    free(A1);
    free(B1);
    free(P_cpu1);
    free(P_gpu1);
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_P1);

    // Free memory for experiment 2
    free(A2);
    free(B2);
    free(P_cpu2);
    free(P_gpu2);
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_P2);

    return 0;
}
