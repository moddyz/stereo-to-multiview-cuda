#ifndef D_FILTER_H
#define D_FILTER_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__host__ __device__ float gaussian2D(float x, float y, float sigma);

void generateGaussianKernel(float* kernel, int radius, float sigma);

__global__ void filter_bleed_1_kernel(unsigned char *img_out, unsigned char *img_in,
                                      int radius, int kernel_sz,
                                      int num_rows, int num_cols);

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

void d_filter_bleed_1(unsigned char *d_img,
                      int radius,
                      int num_rows, int num_cols);

void filter_bleed_1(unsigned char *img,
                    int radius,
                    int num_rows, int num_cols);

#endif
