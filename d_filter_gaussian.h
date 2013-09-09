#ifndef D_FILTER_GAUSSIAN_H
#define D_FILTER_GAUSSIAN_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void filter_gaussian_1_kernel_1(float* img_out, float* img_in,
                                          float *kernel,
                                          int radius, float sigma_spatial,
                                          int num_rows, int num_cols,
                                          int sm_img_rows, int sm_img_cols, int sm_img_sz, int sm_img_padding,
                                          int sm_kernel_len, int sm_kernel_sz);

__global__ void filter_gaussian_1_kernel(float* img_out, float* img_in,
                                       float *kernel,
                                       float sigma_spatial, int radius,
                                       int num_rows, int num_cols);

void filter_gaussian_1(float *img,
                        int radius, float sigma_spatial,
                        int num_rows, int num_cols);

void d_filter_gaussian_1(float *d_img,
                          int radius, float sigma_spatial,
                          int num_rows, int num_cols);

__host__ __device__ float gaussian2D(float x, float y, float sigma);

void generateGaussianKernel(float* kernel, int radius, float sigma);

#endif
