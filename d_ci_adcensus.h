#ifndef D_CI_ADCENSUS_H
#define D_CI_ADCENSUS_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void ci_adcensus(unsigned char* img_l, unsigned char* img_r, float** cost_l, float** cost_r, 
                 float ad_coeff, float census_coeff, int num_disp, int zero_disp, 
                 int num_rows, int num_cols, int elem_sz);

#endif
