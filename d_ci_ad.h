#ifndef D_CI_AD_H
#define D_CI_AD_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void ci_ad_kernel_6(unsigned char* img_l, unsigned char* img_r, 
                                float** cost_l, float** cost_r,
                                int num_disp, int zero_disp, 
                                int num_rows, int num_cols,
                                int sm_cols, int sm_sz, int sm_padding);

__global__ void ci_ad_kernel_5(unsigned char* img_l, unsigned char* img_r, 
                                float** cost_l, float** cost_r,
                                int num_disp, int zero_disp, 
                                int num_rows, int num_cols, int elem_sz,
                                int sm_cols, int sm_sz, int sm_padding);

__global__ void ci_ad_kernel_4(unsigned char* img_l, unsigned char* img_r, 
                                float** cost_l, float** cost_r,
                                int num_disp, int zero_disp, 
                                int num_rows, int num_cols, int elem_sz,
                                int sm_cols, int sm_sz, int sm_padding);

__global__ void ci_ad_kernel_3(unsigned char* img_l, unsigned char* img_r, 
                                float** cost_l, float** cost_r,
                                int num_disp, int zero_disp, 
                                int num_rows, int num_cols, int elem_sz);


__global__ void ci_ad_kernel_2(unsigned char* img_l, unsigned char* img_r, 
                                float** cost_l, float** cost_r,
                                int num_disp, int zero_disp, 
                                int num_rows, int num_cols, int elem_sz);

__global__ void ci_ad_kernel(unsigned char* img_l, unsigned char* img_r, 
                             float** cost, 
                             int num_disp, int zero_disp, int dir,
                             int num_rows, int num_cols, int elem_sz);

#endif
