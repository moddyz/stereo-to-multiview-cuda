#ifndef D_CA_CROSS_SUM_H
#define D_CA_CROSS_SUM_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void cost_copy_kernel(float **cost, float** cost_t, 
                                 int num_disp, int num_rows, int num_cols);

__global__ void cost_transpose_kernel(float **cost, float** cost_t, 
                                      int num_disp, int num_rows, int num_cols);

__global__ void ca_cross_vhsum_kernel(float** cost, float** acost, unsigned char** cross,
                                       int num_disp, int num_rows, int num_cols,
                                       int sm_cols, int sm_sz, int sm_padding);

__global__ void ca_cross_hsum_kernel_2(float** cost, float** acost, unsigned char** cross,
                                       int num_disp, int num_rows, int num_cols,
                                       int sm_cols, int sm_sz, int sm_padding);

__global__ void ca_cross_hsum_kernel(float** cost, float** acost, unsigned char** cross,
                                     int num_disp, int num_rows, int num_cols);

__global__ void ca_cross_vsum_kernel_2(float** cost, float** acost, unsigned char** cross,
                                       int num_disp, int num_rows, int num_cols,
                                       int sm_rows, int sm_sz, int sm_padding);

__global__ void ca_cross_vsum_kernel(float** cost, float** acost, unsigned char** cross,
                                     int num_disp, int num_rows, int num_cols);
#endif
