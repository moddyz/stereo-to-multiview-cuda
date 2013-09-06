#ifndef D_DIBR_FWARP_H
#define D_DIBR_FWARP_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void dibr_forward_warp_kernel(unsigned char* img_out, unsigned char* occl, 
                                         unsigned char* img_in, float* disp,
                                         float shift, int num_rows, int num_cols, int elem_sz);

void d_dibr_dfm(unsigned char* d_img_out,
                unsigned char* d_img_in_l, unsigned char* d_img_in_r, 
                float* disp_l, float* disp_r,
                float shift, int num_rows, int num_cols, int elem_sz);

void dibr_dfm(unsigned char* img_out,
              unsigned char* img_in_l, unsigned char* img_in_r, 
              float* disp_l, float* disp_r,
              float shift, int num_rows, int num_cols, int elem_sz);

#endif
