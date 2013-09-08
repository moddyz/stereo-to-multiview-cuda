#ifndef D_CI_CENSUS_KERNEL 
#define D_CI_CENSUS_KERNEL
#include "d_ci_census.h"
#include "d_alu.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void tx_census_9x7_kernel(unsigned char* img, unsigned long long* census, 
                                     int num_rows, int num_cols, int elem_sz)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
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
            int cx = min(max(gx + x, 0), num_cols - 1);
            int cy = min(max(gy + y, 0), num_rows - 1);
            if (x != 0 && y != 0)
            {
                cb = cb << 1;
                cg = cg << 1;
                cr = cr << 1;
                if (img[(cx + cy * num_cols) * elem_sz] < img[(gx + gy * num_cols) * elem_sz])
                    cb = cb + 1;
                if (img[(cx + cy * num_cols) * elem_sz + 1] < img[(gx + gy * num_cols) * elem_sz + 1])
                    cg = cg + 1;
                if (img[(cx + cy * num_cols) * elem_sz + 2] < img[(gx + gy * num_cols) * elem_sz + 2])
                    cr = cr + 1;
            }
        }
    }
    census[(gx + gy * num_cols) * elem_sz] = cb;
    census[(gx + gy * num_cols) * elem_sz + 1] = cg;
    census[(gx + gy * num_cols) * elem_sz + 2] = cr;
}

// Left & Right Calculation into 1 Module
__global__ void ci_census_kernel_3(unsigned long long *census_l, unsigned long long *census_r, 
                                  float **cost_l, float **cost_r,
                                  int num_disp, int zero_disp,
                                  int num_rows, int num_cols, int elem_sz,
                                  int sm_cols, int sm_sz, 
                                  int sm_padding_l, int sm_padding_r)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ unsigned long long sm_census[];
    unsigned long long* sm_census_l = sm_census;
    unsigned long long* sm_census_r = sm_census + sm_sz;

    int ty_sm_cols = ty * sm_cols;
    int gy_num_cols = gy * num_cols;

    // Load Shared Memory
    for (int elem_offset = 0; elem_offset < elem_sz; ++elem_offset)
    {
        for (int gsx_l = gx - sm_padding_r, gsx_r = gx - sm_padding_l, tsx = tx; 
             tsx < sm_cols; gsx_l += blockDim.x, gsx_r += blockDim.x, tsx += blockDim.x)
        {
            int sm_idx = tsx + ty_sm_cols;
            int gm_idx_l = (min(max(gsx_l, 0), num_cols - 1) + gy_num_cols) * elem_sz + elem_offset;
            int gm_idx_r = (min(max(gsx_r, 0), num_cols - 1) + gy_num_cols) * elem_sz + elem_offset;
        
            sm_census_l[sm_idx] = census_l[gm_idx_l];
            sm_census_r[sm_idx] = census_r[gm_idx_r];
        }
        __syncthreads();

        for (int d = 0; d < num_disp; ++d)
        {
            int r_offset = tx + sm_padding_l + (d - zero_disp);
            int l_idx = tx + sm_padding_r + ty_sm_cols;
            int r_idx = r_offset + ty_sm_cols;

            float cost_1 = (float) alu_hamdist_64(sm_census_l[l_idx], sm_census_r[r_idx]);
            cost_l[d][gx + gy_num_cols] += cost_1 * 0.33333333333f;
            
            int l_offset = tx + sm_padding_r - (d - zero_disp);
            l_idx = tx + sm_padding_l + ty_sm_cols;
            r_idx = l_offset + ty_sm_cols;
            
            cost_1 = (float) alu_hamdist_64(sm_census_r[l_idx], sm_census_l[r_idx]);
            cost_r[d][gx + gy_num_cols] += cost_1 * 0.33333333333f;
        }
        __syncthreads();
    }
}

// Left & Right Calculation into 1 Module
__global__ void ci_census_kernel_2(unsigned long long *census_l, unsigned long long *census_r, 
                                  float **cost_l, float **cost_r,
                                  int num_disp, int zero_disp,
                                  int num_rows, int num_cols, int elem_sz)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int gy_num_cols = gy * num_cols;

    int l_idx = (gx + gy_num_cols) * elem_sz;
    for (int d = 0; d < num_disp; ++d)
    {
        int r_offset = min(max(gx + (d - zero_disp), 0), num_cols - 1);
        int r_idx = (r_offset + gy_num_cols) * elem_sz;

        float cost_1 = (float) alu_hamdist_64(census_l[l_idx], census_r[r_idx]);
        float cost_2 = (float) alu_hamdist_64(census_l[l_idx + 1], census_r[r_idx + 1]);
        float cost_3 = (float) alu_hamdist_64(census_l[l_idx + 2], census_r[r_idx + 2]);
        
        float cost_average = (cost_1 + cost_2 + cost_3) * 0.33333333333f;
        cost_l[d][gx + gy_num_cols] = cost_average;
        
        int l_offset = min(max(gx - (d - zero_disp), 0), num_cols - 1);
        r_idx = (l_offset + gy_num_cols) * elem_sz;
        
        cost_1 = (float) alu_hamdist_64(census_r[l_idx], census_l[r_idx]);
        cost_2 = (float) alu_hamdist_64(census_r[l_idx + 1], census_l[r_idx + 1]);
        cost_3 = (float) alu_hamdist_64(census_r[l_idx + 2], census_l[r_idx + 2]);
        
        cost_average = (cost_1 + cost_2 + cost_3) * 0.33333333333f;
        cost_r[d][gx + gy_num_cols] = cost_average;
    }
}

__global__ void ci_census_kernel(unsigned long long* census_l, unsigned long long* census_r, 
                                 float** cost_l,
                                 int num_disp, int zero_disp, int dir,
                                 int num_rows, int num_cols, int elem_sz)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int gy_num_cols = gy * num_cols;

    for (int d = 0; d < num_disp; ++d)
    {
        int r_coord = min(max(gx + dir * (d - zero_disp), 0), num_cols - 1);
        int l_idx = (gx + gy_num_cols) * elem_sz;
        int r_idx = (r_coord + gy_num_cols) * elem_sz;
        float cost_b = (float) alu_hamdist_64(census_l[l_idx], census_r[r_idx]);
        float cost_g = (float) alu_hamdist_64(census_l[l_idx + 1], census_r[r_idx + 1]);
        float cost_r = (float) alu_hamdist_64(census_l[l_idx + 2], census_r[r_idx + 2]);
        float cost = (cost_b + cost_g + cost_r) * 0.33333333333;
        cost_l[d][gx + gy_num_cols] = cost;
    }
}


#endif
