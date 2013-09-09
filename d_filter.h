#ifndef D_FILTER_H
#define D_FILTER_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


__global__ void filter_bleed_1_kernel(unsigned char *img_out, unsigned char *img_in,
                                      int radius, int kernel_sz,
                                      int num_rows, int num_cols);


void d_filter_bleed_1(unsigned char *d_img,
                      int radius,
                      int num_rows, int num_cols);

void filter_bleed_1(unsigned char *img,
                    int radius,
                    int num_rows, int num_cols);

#endif
