#ifndef D_DIBR_WARP_H
#define D_DIBR_WARP_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void dibr_forward_map_kernel(unsigned char* img_out, unsigned char* img_in, int* disp_in,
                                        float shift, int num_disp, int zero_disp, 
                                        int num_rows, int num_cols, int elem_sz);

void dibr_forwardmerge(unsigned char* img_out_l, unsigned char* img_out_r, 
                       unsigned char* img_in_l, unsigned char* img_in_r, int* disp_l, int* disp_r,
                       float shift, int num_disp, int zero_disp, 
                       int num_rows, int num_cols, int elem_sz);

#endif
