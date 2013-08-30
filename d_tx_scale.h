#ifndef D_TX_SCALE_H
#define D_TX_SCALE_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "d_alu.h"

__global__ void tx_scale_bilinear_kernel(unsigned char* img_in, unsigned char* img_out, 
                                         int in_rows, int in_cols, int out_rows, int out_cols, int elem_sz);

__global__ void tx_scale_nearest_kernel(unsigned char* img_in, unsigned char* img_out, 
                                        int in_rows, int in_cols, int out_rows, int out_cols, int elem_sz);

void d_tx_scale(unsigned char* in_data, unsigned char* out_data, 
                int in_rows, int in_cols, int out_rows, int out_cols, int elem_sz);

#endif

