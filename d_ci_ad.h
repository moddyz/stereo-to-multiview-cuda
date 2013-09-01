#ifndef D_CI_AD_H
#define D_CI_AD_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


__global__ void ci_ad_kernel(unsigned char* img_l, unsigned char* img_r, 
                             float** cost_l, float** cost_R, int num_disp, int zero_disp,
                             int num_rows, int num_cols, int elem_sz,
                             int sm_w, int sm_sz);

void ci_ad(unsigned char* img_l, unsigned char* img_r, float** cost_l, float** cost_r, 
           int num_disp, int zero_disp, int num_rows, int num_cols, int elem_sz);

#endif
