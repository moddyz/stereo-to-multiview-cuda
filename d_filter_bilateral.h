#ifndef D_FILTER_BILATERAL_H
#define D_FILTER_BILATERAL_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void filter_bilateral_1_kernel_5(float *img_out, float* kernel,
                                            int radius, float sigma_color, float sigma_color_sqrt_pi,
                                            int num_rows, int num_cols,
                                            int sm_kernel_len, int sm_kernel_sz);

__global__ void filter_bilateral_1_kernel_4(float *img_out, float* kernel,
                                            int radius, float sigma_color, float sigma_color_sqrt_pi,
                                            int num_rows, int num_cols);

void filter_bilateral_1_tex(float *img,
                            int radius, float sigma_color, float sigma_spatial,
                            int num_rows, int num_cols);

__global__ void filter_bilateral_1_kernel_3(float *img_out, float *img_in, float* kernel,
                                            int radius, 
                                            float sigma_color, float sigma_color_sqrt_pi, float sigma_spatial,
                                            int num_rows, int num_cols,
                                            int sm_img_rows, int sm_img_cols, int sm_img_sz, int sm_img_padding,
                                            int sm_kernel_len, int sm_kernel_sz);

__global__ void filter_bilateral_1_kernel_2(float *img_out, float *img_in, float* kernel,
                                            int radius, float sigma_color, float sigma_spatial,
                                            int num_rows, int num_cols,
                                            int sm_img_rows, int sm_img_cols, int sm_img_sz, int sm_img_padding,
                                            int sm_kernel_len, int sm_kernel_sz);

__global__ void filter_bilateral_1_kernel(float *img_out, float *img_in, float* kernel,
                                          int radius, float sigma_color, float sigma_spatial,
                                          int num_rows, int num_cols);

void d_filter_bilateral_1(float *d_img,
                          int radius, float sigma_color, float sigma_spatial,
                          int num_rows, int num_cols);

void filter_bilateral_1(float *img,
                        int radius, float sigma_color, float sigma_spatial,
                        int num_rows, int num_cols);

#endif
