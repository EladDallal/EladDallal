// Elad Dallal (20264650)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <ctime>

#define S_100 100
#define S_250 250
#define S_500 500
#define S_1000 1000
#define S_1500 1500

#define BLOCK_SIZES {1, 2, 5, 10, 25, 32}

int flag = 0; //flag for checking if matrices are =

// Device matrix multiplication calculates row and col of the grid / block and then
// flattens matrix before inserting values
__global__ void DeviceMatrixMultiplication(int* A, int* B, int* O, int size) {
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

void HostMatrixMultiplication(int* A, int* B, int* C, int size) {
    int offset1, offset2;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float temp = 0;
            for (int k = 0; k < size; k++) {
                offset1 = i * size + k;
                offset2 = k * size + j;

                temp += A[offset1] * B[offset2];
            }
            C[i * size + j] = temp;
        }
    }
}

int main() {
    // Sizes of matrices
    const int sizes[] = { S_100, S_250, S_500, S_1000, S_1500 };
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Block sizes
    const int block_sizes[] = { 1, 2, 5, 10, 25, 32 };
    const int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

    for (int size_index = 0; size_index < num_sizes; ++size_index) {
        int size = sizes[size_index];
        size_t hostSize = size * size * sizeof(int);

        // Allocate host memory
        int* h_A = (int*)malloc(hostSize);
        int* h_B = (int*)malloc(hostSize);
        int* h_C = (int*)malloc(hostSize);
        int* h_P = (int*)malloc(hostSize);

        // Initialize host matrices
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // Get the 2 random values and assign
                int rand1 = rand() % 10;
                int rand2 = rand() % 10;
                *(h_A + i * size + j) = rand1;
                *(h_B + i * size + j) = rand2;
            }
        }

        // Allocate device memory
        int* d_A = nullptr;
        int* d_B = nullptr;
        int* d_C = nullptr;
        cudaMalloc((void**)&d_A, hostSize);
        cudaMalloc((void**)&d_B, hostSize);
        cudaMalloc((void**)&d_C, hostSize);

        for (int block_size_index = 0; block_size_index < num_block_sizes; ++block_size_index) {
            int block_size = block_sizes[block_size_index];
            dim3 threadsPerBlock(block_size, block_size);
            dim3 numberOfBlocks(ceil(size / (float)threadsPerBlock.x), ceil(size / (float)threadsPerBlock.y), 1);

            // Time the host-to-device transfer
            cudaEvent_t start_transfer_hd, stop_transfer_hd;
            cudaEventCreate(&start_transfer_hd);
            cudaEventCreate(&stop_transfer_hd);
            cudaEventRecord(start_transfer_hd, 0);
            cudaMemcpy(d_A, h_A, hostSize, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, hostSize, cudaMemcpyHostToDevice);
            cudaEventRecord(stop_transfer_hd, 0);
            cudaEventSynchronize(stop_transfer_hd);
            float transfer_hd_time;
            cudaEventElapsedTime(&transfer_hd_time, start_transfer_hd, stop_transfer_hd);

            // Time the kernel execution
            cudaEvent_t start_kernel, stop_kernel;
            cudaEventCreate(&start_kernel);
            cudaEventCreate(&stop_kernel);
            cudaEventRecord(start_kernel, 0);
            DeviceMatrixMultiplication << <numberOfBlocks, threadsPerBlock >> > (d_A, d_B, d_C, size);
            cudaEventRecord(stop_kernel, 0);
            cudaEventSynchronize(stop_kernel);
            float kernel_time;
            cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

            // Time the device-to-host transfer
            cudaEvent_t start_transfer_dh, stop_transfer_dh;
            cudaEventCreate(&start_transfer_dh);
            cudaEventCreate(&stop_transfer_dh);
            cudaEventRecord(start_transfer_dh, 0);
            cudaMemcpy(h_C, d_C, hostSize, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop_transfer_dh, 0);
            cudaEventSynchronize(stop_transfer_dh);
            float transfer_dh_time;
            cudaEventElapsedTime(&transfer_dh_time, start_transfer_dh, stop_transfer_dh);

            // Compute the CPU multiplication time
            clock_t cpu_start = clock();
            HostMatrixMultiplication(h_A, h_B, h_P, size);
            clock_t cpu_end = clock();
            float cpu_time = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

            // Print results
            printf("Matrix Size: %dx%d, Block Size: %d\n", size, size, block_size);
            printf("Host to Device Transfer Time: %0.2f ms\n", transfer_hd_time);
            printf("Kernel Multiplication Time: %0.2f ms\n", kernel_time);
            printf("Device to Host Transfer Time: %0.2f ms\n", transfer_dh_time);
            printf("CPU Multiplication Time: %0.2f ms\n", cpu_time);
            printf("\n");
        }

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Free host memory
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_P);
    }

    return 0;
}
