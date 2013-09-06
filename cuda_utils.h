#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <stdio.h>

#define checkCudaError(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

struct cudaEventPair_t
{
    cudaEvent_t start;
    cudaEvent_t end;
};


inline void startCudaTimer(cudaEventPair_t * p)
{
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}

inline void stopCudaTimer(cudaEventPair_t * p, char *message)
{
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, p->start, p->end);
    printf("[[ %s took: %.4f ms ]]\n\n", message, elapsedTime);

    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
}

inline void printDeviceInfo()
{
    printf("=======================\n");
    printf("== CUDA DEVICE QUERY ==\n");
    printf("=======================\n\n");
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; ++i)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("Device Number:                 %d\n",  i);
        printf("Major revision number:         %d\n",  devProp.major);
        printf("Minor revision number:         %d\n",  devProp.minor);
        printf("Name:                          %s\n",  devProp.name);
        printf("Total global memory:           %u\n", (unsigned int) devProp.totalGlobalMem);
        printf("Total shared memory per block: %u\n",  (unsigned int)devProp.sharedMemPerBlock);
        printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
        printf("Warp size:                     %d\n",  devProp.warpSize);
        printf("Maximum memory pitch:          %u\n",  (unsigned int)devProp.memPitch);
        printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
        for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("Clock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %u\n",  (unsigned int) devProp.totalConstMem);
        printf("Texture alignment:             %u\n",  (unsigned int) devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
        printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
        printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
		printf("\n");
    }
}

#endif
