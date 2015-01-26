#ifndef D_DR_IRV_H
#define D_DR_IRV_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>



void dr_interp(float* disp, unsigned char* outliers, 
               int num_rows, int num_cols, int num_disp, int zero_disp);

#endif
