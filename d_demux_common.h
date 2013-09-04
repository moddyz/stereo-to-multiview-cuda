#ifndef D_DEMUX_COMMON_H
#define D_DEMUX_COMMON_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void demux_rgb(unsigned char* chan_r, unsigned char* chan_g, unsigned char* chan_b,
                          unsigned char* img, 
                          int num_rows, int num_cols, int elem_sz);

#endif
