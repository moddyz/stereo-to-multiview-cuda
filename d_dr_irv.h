#ifndef D_DR_IRV_H
#define D_DR_IRV_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void dr_irv( float* disp, unsigned char* outliers, unsigned char **cross,
             int thresh_s, float thresh_h,
             int num_rows, int num_cols, int num_disp, int zero_disp,
             int usd,
             int iterations);

#endif
