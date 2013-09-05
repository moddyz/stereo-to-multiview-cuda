#ifndef D_CI_ADCENSUS_H
#define D_CI_ADCENSUS_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void ci_adcensus_kernel(float** ad_cost_l, float** ad_cost_r, 
                                   float** census_cost_l, float** census_cost_r,
                                   float** adcensus_cost_l, float** adcensus_cost_r,
                                   float inv_ad_coeff, float inv_census_coeff, 
                                   int num_disp, int zero_disp, 
                                   int num_rows, int num_cols, int elem_sz);

void d_ci_adcensus(unsigned char* d_img_l, unsigned char* d_img_r, 
                 float** d_adcensus_cost_l, float** d_adcensus_cost_r, 
                 float** h_adcensus_cost_l, float** h_adcensus_cost_r, 
                 float ad_coeff, float census_coeff, int num_disp, int zero_disp, 
                 int num_rows, int num_cols, int elem_sz);

void ci_adcensus(unsigned char* img_l, unsigned char* img_r, float** cost_l, float** cost_r, 
                 float ad_coeff, float census_coeff, int num_disp, int zero_disp, 
                 int num_rows, int num_cols, int elem_sz);

#endif
