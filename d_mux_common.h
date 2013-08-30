#ifndef D_MUX_COMMON_H
#define D_MUX_COMMON_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void mux_merge_AB_kernel(unsigned char* img_b, unsigned char* img_a, unsigned char* mask_a,
                                    int num_rows, int num_cols, int elem_sz);

#endif
