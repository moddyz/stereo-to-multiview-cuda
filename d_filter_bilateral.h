#ifndef D_FILTER_BILATERAL_H
#define D_FILTER_BILATERAL_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


void filter_bilateral_1_tex(float *img,
                            int radius, float sigma_color, float sigma_spatial,
                            int num_rows, int num_cols);

void d_filter_bilateral_1(float *d_img,
                          int radius, float sigma_color, float sigma_spatial,
                          int num_rows, int num_cols, int num_disp);

void filter_bilateral_1(float *img,
                        int radius, float sigma_color, float sigma_spatial,
                        int num_rows, int num_cols,
                        int num_disp);

#endif
