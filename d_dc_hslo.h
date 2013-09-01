#ifndef D_DC_HSLO_H
#define D_DC_HSLO_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void dc_hslo_h_cdiff_kernel(float* P1_l, float** P2_l, 
                                       float* P1_r, float** P2_r, 
                                       unsigned char* img_l, unsigned char* img_r,
                                       int r_dir,
                                       float T, float H1, float H2, 
                                       float H1_4, float H2_4, 
                                       float H1_10, float H2_10,
                                       int num_disp, int zero_disp, 
                                       int num_rows, int num_cols, int elem_sz);

void dc_hslo(float** cost, float* disp, 
             unsigned char* img_l, unsigned char* img_r,
             float T, float H1, float H2, 
             int num_disp, int zero_disp, 
             int num_rows, int num_cols, int elem_sz);

#endif
