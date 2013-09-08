#ifndef D_CI_CENSUS_KERNEL 
#define D_CI_CENSUS_KERNEL
#include "d_ci_census.h"
#include "d_alu.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void tx_census_9x7_kernel(unsigned char* img, unsigned long long* census, 
                                     int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;
    
    int win_h2 = 3; // Half of 7 + anchor
    int win_w2 = 4; // Half of 9 + anchor

    unsigned long long cb = 0;
    unsigned long long cg = 0;
    unsigned long long cr = 0;
    
    for (int y = -win_h2; y <= win_h2; ++y)
    {
        for (int x = -win_w2; x <= win_w2; ++x)
        {
            int cx = min(max(tx + x, 0), num_cols - 1);
            int cy = min(max(ty + y, 0), num_rows - 1);
            if (x != 0 && y != 0)
            {
                cb = cb << 1;
                cg = cg << 1;
                cr = cr << 1;
                if (img[(cx + cy * num_cols) * elem_sz] < img[(tx + ty * num_cols) * elem_sz])
                    cb = cb + 1;
                if (img[(cx + cy * num_cols) * elem_sz + 1] < img[(tx + ty * num_cols) * elem_sz + 1])
                    cg = cg + 1;
                if (img[(cx + cy * num_cols) * elem_sz + 2] < img[(tx + ty * num_cols) * elem_sz + 2])
                    cr = cr + 1;
            }
        }
    }
    census[(tx + ty * num_cols) * elem_sz] = cb;
    census[(tx + ty * num_cols) * elem_sz + 1] = cg;
    census[(tx + ty * num_cols) * elem_sz + 2] = cr;
}

__global__ void ci_census_kernel(unsigned long long* census_l, unsigned long long* census_r, 
                                 float** cost_l,
                                 int num_disp, int zero_disp, int dir,
                                 int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;

    int ty_num_cols = ty * num_cols;

    for (int d = 0; d < num_disp; ++d)
    {
        int r_coord = min(max(tx + dir * (d - zero_disp), 0), num_cols - 1);
        int l_idx = (tx + ty_num_cols) * elem_sz;
        int r_idx = (r_coord + ty_num_cols) * elem_sz;
        float cost_b = (float) alu_hamdist_64(census_l[l_idx], census_r[r_idx]);
        float cost_g = (float) alu_hamdist_64(census_l[l_idx + 1], census_r[r_idx + 1]);
        float cost_r = (float) alu_hamdist_64(census_l[l_idx + 2], census_r[r_idx + 2]);
        float cost = (cost_b + cost_g + cost_r) * 0.33333333333;
        cost_l[d][tx + ty_num_cols] = cost;
    }
}


#endif
