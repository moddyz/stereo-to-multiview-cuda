#ifndef D_FILTER_H
#define D_FILTER_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void filter_bilateral_1_kernel(float *img_out, float *img_in, float* kernel,
                                   int radius, float sigma_color, float sigma_spatial,
                                   int num_rows, int num_cols);

void d_filter_bilateral_1(float *d_img_out, float *d_img_in,
                          int radius, float sigma_color, float sigma_spatial,
                          int num_rows, int num_cols);

void filter_bilateral_1(float *img_out, float *img_in,
                        int radius, float sigma_color, float sigma_spatial,
                        int num_rows, int num_cols);

#endif
