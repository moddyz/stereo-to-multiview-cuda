#ifndef D_CA_CROSS_H
#define D_CA_CROSS_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "d_ca_cross_sum.h"

__global__ void ca_cross_construction_kernel(unsigned char* img, unsigned char** cross,
                                             float ucd, float lcd, int usd, int lsd,
                                             int num_rows, int num_cols, int elem_sz);

void d_ca_cross(unsigned char* d_img, float** d_cost, 
                float** d_acost, float** h_acost, float* d_acost_memory,
                unsigned char **d_cross, 
                float ucd, float lcd, int usd, int lsd,
                int num_disp, int num_rows, int num_cols, int elem_sz);

void ca_cross(unsigned char* img, unsigned char **cross, float** cost, float** acost,
              float ucd, float lcd, int usd, int lsd,
              int num_disp, int num_rows, int num_cols, int elem_sz);

#endif
