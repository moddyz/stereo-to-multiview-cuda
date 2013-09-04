#ifndef D_DEMUX_COMMON_KERNEL 
#define D_DEMUX_COMMON_KERNEL
#include "d_demux_common.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void demux_rgb(unsigned char* chan_r, unsigned char* chan_g, unsigned char* chan_b,
                          unsigned char* img, 
                          int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tx > num_cols * num_rows - 1)
        return;

    int tx_e = tx * elem_sz;
    chan_r[tx] = img[tx_e + 2];
    chan_g[tx] = img[tx_e + 1];
    chan_b[tx] = img[tx_e];
}

#endif
