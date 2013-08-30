#ifndef D_ALU_KERNEL
#define D_ALU_KERNEL
#include "d_alu.h"
#include "cuda_utils.h"
#include <math.h>

__device__ int alu_hamdist_64(unsigned long long a, unsigned long long b)
{
    int c = a ^ b;
    unsigned long long mask = 1;
    int dist = 0;
    for (int i = 0; i < 64; ++i)
    {
        if (c & mask == 1)
            ++dist;
        c = c >> 1;
    }
    return dist;
}

__device__ unsigned int alu_bilinear_interp(unsigned char* data, int elem_sz, int elem_offset, float coord_x, float coord_y, int width, int height) 
{
    int coord_00_x = floor(coord_x);
    int coord_00_y = floor(coord_y);
    
    int coord_01_x = min(coord_00_x + 1, width - 1);
    int coord_01_y = coord_00_y;

    int coord_10_x = coord_00_x;
    int coord_10_y = min(coord_00_y + 1, height - 1);
    
    int coord_11_x = min(coord_00_x + 1, width - 1);
    int coord_11_y = min(coord_00_y + 1, height - 1);

    float weight_x = coord_x - (float) coord_00_x;
    float weight_y = coord_y - (float) coord_00_y;

    unsigned char val_00 = data[(coord_00_x + coord_00_y * width) * elem_sz + elem_offset];
    unsigned char val_01 = data[(coord_01_x + coord_01_y * width) * elem_sz + elem_offset];
    unsigned char val_10 = data[(coord_10_x + coord_10_y * width) * elem_sz + elem_offset];
    unsigned char val_11 = data[(coord_11_x + coord_11_y * width) * elem_sz + elem_offset];

    float top = (float) val_00 * (1.0 - weight_x) + (float) val_01 * weight_x;
    float bot = (float) val_10 * (1.0 - weight_x) + (float) val_11 * weight_x;
    
    return (unsigned char) (top * (1.0 - weight_y) + bot * weight_y);
}


#endif
