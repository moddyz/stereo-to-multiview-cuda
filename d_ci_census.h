#ifndef D_CI_CENSUS_H
#define D_CI_CENSUS_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void tx_census_9x7_kernel(unsigned char* img, unsigned long long* census, 
                                     int num_rows, int num_cols, int elem_sz);

__global__ void ci_census_kernel(unsigned long long* census_l, unsigned long long* census_r, 
                                 float** cost_l,
                                 int num_disp, int zero_disp, int dir,
                                 int num_rows, int num_cols, int elem_sz);

#endif
