#ifndef D_DEMUX_COMMON_KERNEL 
#define D_DEMUX_COMMON_KERNEL
#include "d_demux_common.h"
#include "cuda_utils.h"
#include <math.h>


__global__ void demux_sbs(unsigned char* img_l, unsigned char* img_r,
                          unsigned char* img_sbs,
                          int num_rows, int num_cols_sbs, int num_cols_out, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx > num_cols_sbs - 1 || ty > num_rows - 1)
        return;

    if (tx < num_cols_out)
    {
        int sx = tx;
        img_l[(sx + ty * num_cols_out) * elem_sz] = img_sbs[(tx + ty * num_cols_sbs) * elem_sz];
        img_l[(sx + ty * num_cols_out) * elem_sz + 1] = img_sbs[(tx + ty * num_cols_sbs) * elem_sz + 1];
        img_l[(sx + ty * num_cols_out) * elem_sz + 2] = img_sbs[(tx + ty * num_cols_sbs) * elem_sz + 2];
    }
    else 
    {
        int sx = tx - num_cols_out;
        img_r[(sx + ty * num_cols_out) * elem_sz] = img_sbs[(tx + ty * num_cols_sbs) * elem_sz];
        img_r[(sx + ty * num_cols_out) * elem_sz + 1] = img_sbs[(tx + ty * num_cols_sbs) * elem_sz + 1];
        img_r[(sx + ty * num_cols_out) * elem_sz + 2] = img_sbs[(tx + ty * num_cols_sbs) * elem_sz + 2];
    }

}

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

#end
