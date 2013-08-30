#ifndef D_ALU_H
#define D_ALU_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

extern __device__ int alu_hamdist_64(unsigned long long a, unsigned long long b);

extern __device__ unsigned int alu_bilinear_interp(unsigned char* data, int elem_sz, int elem_offset, float coord_x, float coord_y, int width, int height);

#endif
