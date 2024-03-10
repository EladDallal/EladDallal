//#include <stdio.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <string.h>
//
//
//int _ConvertSMVer2Cores(int major, int minor) {
//    typedef struct {
//        int SM;  
//        int Cores;  
//    } SMInfo;
//
//    SMInfo info[] = {
//        {0x30, 192},
//        {0x32, 192},
//        {0x35, 192},
//        {0x37, 192},
//        {0x50, 128},
//        {0x52, 128},
//        {0x53, 128},
//        {0x60, 64},
//        {0x61, 128},
//        {0x62, 128},
//        {0x70, 64},
//        {0x72, 64},
//        {0x75, 64},
//        {0x80, 64},
//        {0x86, 64},
//        {0x86, 64}
//    };
//
//    int size = sizeof(info) / sizeof(info[0]);
//
//    for (int i = 0; i < size; ++i) {
//        if (info[i].SM == ((major << 4) + minor)) {
//            return info[i].Cores;
//        }
//    }
//    return -1;
//}
//
//int main() {
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//
//    printf("Number of CUDA devices: %d\n", deviceCount);
//
//    for (int i = 0; i < deviceCount; ++i) {
//        cudaDeviceProp prop;
//        cudaGetDeviceProperties(&prop, i);
//
//        printf("\nDevice %d:\n", i);
//        printf("  Name: %s\n", prop.name);
//        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
//        printf("  Clock rate: %d kHz\n", prop.clockRate);
//        printf("  Number of streaming multiprocessors (SM): %d\n", prop.multiProcessorCount);
//        printf("  Number of cores per SM: %d\n", _ConvertSMVer2Cores(prop.major, prop.minor));
//        printf("  Warp size: %d\n", prop.warpSize);
//        printf("  Global memory: %.2f MB\n", (float)prop.totalGlobalMem / (1024 * 1024));
//        printf("  Constant memory: %.2f KB\n", (float)prop.totalConstMem / 1024);
//        printf("  Shared memory per block: %.2f KB\n", (float)prop.sharedMemPerBlock / 1024);
//        printf("  Registers available per block: %d\n", prop.regsPerBlock);
//        printf("  Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
//        printf("  Maximum size of each dimension of a block: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
//        printf("  Maximum size of each dimension of a grid: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
//    }
//
//    return 0;
//}