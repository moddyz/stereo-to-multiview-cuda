#ifndef D_DIBR_WARP_H
#define D_DIBR_WARP_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void dibr_forward_warp_kernel(unsigned char* img_out, unsigned char* holes, 
                                         unsigned char* img_in, float* disp,
                                         float shift, int num_rows, int num_cols, int elem_sz);

void dibr_dfm(unsigned char* img_out,
              unsigned char* img_in_l, unsigned char* img_in_r, float* disp_l, float* disp_r,
              float shift, int num_rows, int num_cols, int elem_sz);

#endif
