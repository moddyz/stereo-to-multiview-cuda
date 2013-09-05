#ifndef D_DEMUX_COMMON_H
#define D_DEMUX_COMMON_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


__global__ void demux_sbs(unsigned char* img_l, unsigned char* img_r,
                          unsigned char* img_sbs,
                          int num_rows, int num_cols_sbs, int num_cols_out, int elem_sz);

__global__ void demux_rgb(unsigned char* chan_r, unsigned char* chan_g, unsigned char* chan_b,
                          unsigned char* img, 
                          int num_rows, int num_cols, int elem_sz);

#endif
