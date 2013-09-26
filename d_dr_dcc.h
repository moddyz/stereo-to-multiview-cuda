#ifndef D_DR_DCC_H
#define D_DR_DCC_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void dr_dcc_kernel(unsigned char* errors_l, unsigned char *errors_r,
                             float *disp_l, float *disp_r,
                             float thresh,
                             int num_rows, int num_cols);

void d_dr_dcc(unsigned char *d_outliers_l, unsigned char *d_outliers_r,
              float* d_disp_l, float *d_disp_r,
              int num_rows, int num_cols);

void dr_dcc(unsigned char *outliers_l, unsigned char *outliers_r,
            float* disp_l, float *disp_r,
            int num_rows, int num_cols);

#endif
