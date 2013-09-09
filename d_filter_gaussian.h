#ifndef D_FILTER_GAUSSIAN_H
#define D_FILTER_GAUSSIAN_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_utils.h"
#include <math.h>

#define PI 3.14159265359

__host__ __device__ float gaussian2D(float x, float y, float sigma);

void generateGaussianKernel(float* kernel, int radius, float sigma);

__global__ void filter_gaussian_1F_kernel(float* img_out, float* img_in,
                                       float *kernel,
                                       float sigma_spatial, int radius,
                                       int num_rows, int num_cols);

void filter_gaussian_1F(float *img,
                        int radius, float sigma_spatial,
                        int num_rows, int num_cols);

void d_filter_gaussian_1F(float *d_img,
                          int radius, float sigma_spatial,
                          int num_rows, int num_cols);

#endif
