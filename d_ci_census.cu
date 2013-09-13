#ifndef D_CI_CENSUS_KERNEL 
#define D_CI_CENSUS_KERNEL
#include "d_ci_census.h"
#include "d_alu.h"
#include "cuda_utils.h"
#include <math.h>

inline __device__ int alu_hamdist_32(unsigned int a, unsigned int b)
{
    int c = a ^ b;
    unsigned int mask = 1;
    int dist = 0;
    for (int i = 0; i < 32; ++i, c >>= 1)
        dist += c & mask;
    return dist;
}

__global__ void tx_census_9x7_kernel_3(unsigned char* img, unsigned long long* census, 
                                       int num_rows, int num_cols)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;
    
    int win_h2 = 3; // Half of 7 + anchor
    int win_w2 = 4; // Half of 9 + anchor

    unsigned long long c = 0;
    int gx_gy_num_cols = gx + gy * num_cols;
    
    unsigned char compare = img[gx_gy_num_cols];
    
    for (int y = -win_h2; y <= win_h2; ++y)
    {
        for (int x = -win_w2; x <= win_w2; ++x)
        {
            int cx = min(max(gx + x, 0), num_cols - 1);
            int cy = min(max(gy + y, 0), num_rows - 1);
            if (x != 0 && y != 0)
            {
                c = c << 1;
                if (img[cx + cy * num_cols] < compare)
                    c = c + 1;
            }
        }
    }
    census[gx_gy_num_cols] = c;
}

__global__ void tx_census_9x7_kernel_2(unsigned char *img, 
                                       unsigned int **census, 
                                       int num_rows, int num_cols, int elem_sz)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;
    
    int win_h2 = 3; // Half of 7 + anchor
    int win_w2 = 4; // Half of 9 + anchor

    unsigned int b1 = 0;
    unsigned int g1 = 0;
    unsigned int r1 = 0;
    
    unsigned int b2 = 0;
    unsigned int g2 = 0;
    unsigned int r2 = 0;

    unsigned int mask = 0xFFFFFF;

    int gx_gy_num_cols_elem_sz = (gx + gy * num_cols) * elem_sz;
    
    for (int y = -win_h2; y <= win_h2; ++y)
    {
        for (int x = -win_w2; x <= win_w2; ++x)
        {
            int cx = min(max(gx + x, 0), num_cols - 1);
            int cy = min(max(gy + y, 0), num_rows - 1);
            if (y < 0)
            {
                b1 = b1 << 1;
                g1 = g1 << 1;
                r1 = r1 << 1;
                
                int cx_cy_num_cols_elem_sz = (cx + cy * num_cols) * elem_sz;

                if (img[cx_cy_num_cols_elem_sz] < img[gx_gy_num_cols_elem_sz])
                    b1 = b1 | 1;
                if (img[cx_cy_num_cols_elem_sz + 1] < img[gx_gy_num_cols_elem_sz + 1])
                    g1 = g1 | 1;
                if (img[cx_cy_num_cols_elem_sz + 2] < img[gx_gy_num_cols_elem_sz + 2])
                    r1 = r1 | 1;
            }
            else if (y == 0)
            {
                if (x < 0)
                {
                    b1 = b1 << 1;
                    g1 = g1 << 1;
                    r1 = r1 << 1;
                    
                    int cx_cy_num_cols_elem_sz = (cx + cy * num_cols) * elem_sz;

                    if (img[cx_cy_num_cols_elem_sz] < img[gx_gy_num_cols_elem_sz])
                        b1 = b1 | 1;
                    if (img[cx_cy_num_cols_elem_sz + 1] < img[gx_gy_num_cols_elem_sz + 1])
                        g1 = g1 | 1;
                    if (img[cx_cy_num_cols_elem_sz + 2] < img[gx_gy_num_cols_elem_sz + 2])
                        r1 = r1 | 1;
                }
                else if (x > 0)
                {
                    b2 = b2 << 1;
                    g2 = g2 << 1;
                    r2 = r2 << 1;
                    
                    int cx_cy_num_cols_elem_sz = (cx + cy * num_cols) * elem_sz;

                    if (img[cx_cy_num_cols_elem_sz] < img[gx_gy_num_cols_elem_sz])
                        b2 = b2 | 1;
                    if (img[cx_cy_num_cols_elem_sz + 1] < img[gx_gy_num_cols_elem_sz + 1])
                        g2 = g2 | 1;
                    if (img[cx_cy_num_cols_elem_sz + 2] < img[gx_gy_num_cols_elem_sz + 2])
                        r2 = r2 | 1;
                }
            }
            else 
            {
                b2 = b2 << 1;
                g2 = g2 << 1;
                r2 = r2 << 1;
                
                int cx_cy_num_cols_elem_sz = (cx + cy * num_cols) * elem_sz;

                if (img[cx_cy_num_cols_elem_sz] < img[gx_gy_num_cols_elem_sz])
                    b2 = b2 | 1;
                if (img[cx_cy_num_cols_elem_sz + 1] < img[gx_gy_num_cols_elem_sz + 1])
                    g2 = g2 | 1;
                if (img[cx_cy_num_cols_elem_sz + 2] < img[gx_gy_num_cols_elem_sz + 2])
                    r2 = r2 | 1;
            }
        }
    }
    census[0][gx_gy_num_cols_elem_sz] = b1;
    census[0][gx_gy_num_cols_elem_sz + 1] = g1;
    census[0][gx_gy_num_cols_elem_sz + 2] = r1;
    
    census[1][gx_gy_num_cols_elem_sz] = b2;
    census[1][gx_gy_num_cols_elem_sz + 1] = g2;
    census[1][gx_gy_num_cols_elem_sz + 2] = r2;
}


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

__global__ void ci_census_kernel_6(unsigned long long *census_l, unsigned long long *census_r, 
                                  float **cost_l, float **cost_r,
                                  int num_disp, int zero_disp,
                                  int num_rows, int num_cols, 
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
    for (int gsx_l = gx - sm_padding_r, gsx_r = gx - sm_padding_l, tsx = tx; 
         tsx < sm_cols; gsx_l += blockDim.x, gsx_r += blockDim.x, tsx += blockDim.x)
    {
        int sm_idx = tsx + ty_sm_cols;
        int gm_idx_l = min(max(gsx_l, 0), num_cols - 1) + gy_num_cols;
        int gm_idx_r = min(max(gsx_r, 0), num_cols - 1) + gy_num_cols;
    
        sm_census_l[sm_idx] = census_l[gm_idx_l];
        sm_census_r[sm_idx] = census_r[gm_idx_r];
    }
    __syncthreads();

    int l_idx = tx + sm_padding_r + ty_sm_cols;
    int l_idx2 = tx + sm_padding_l + ty_sm_cols;
    unsigned long long l1 = sm_census_l[l_idx];
    unsigned long long r1 = sm_census_r[l_idx2];

    for (int d = 0; d < num_disp; ++d)
    {
        int r_offset = tx + sm_padding_l + (d - zero_disp);
        int r_idx = r_offset + ty_sm_cols;
        int l_offset = tx + sm_padding_r - (d - zero_disp);
        int r_idx2 = l_offset + ty_sm_cols;

        unsigned long long r2 = sm_census_r[r_idx];
        unsigned long long l2 = sm_census_l[r_idx2];
        
        float cost_hamming = (float) alu_hamdist_64(l1, r2);
        cost_l[d][gx + gy_num_cols] += cost_hamming * 0.33333333333f;
        
        cost_hamming = (float) alu_hamdist_64(r1, l2);
        cost_r[d][gx + gy_num_cols] += cost_hamming * 0.33333333333f;
    }
}

__global__ void ci_census_kernel_5(unsigned int **census_l, unsigned int **census_r, 
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

    extern __shared__ unsigned int sm_censusx[];
    unsigned int* sm_census_l_0 = sm_censusx;
    unsigned int* sm_census_l_1 = sm_census_l_0 + sm_sz;
    unsigned int* sm_census_r_0 = sm_census_l_1 + sm_sz;
    unsigned int* sm_census_r_1 = sm_census_r_0 + sm_sz;

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
        
            sm_census_l_0[sm_idx] = census_l[0][gm_idx_l];
            sm_census_l_1[sm_idx] = census_l[1][gm_idx_l];

            sm_census_r_0[sm_idx] = census_r[0][gm_idx_r];
            sm_census_r_1[sm_idx] = census_r[1][gm_idx_r];
        }
        __syncthreads();

        int l_idx = tx + sm_padding_r + ty_sm_cols;
        int l_idx2 = tx + sm_padding_l + ty_sm_cols;
        unsigned int l1_1 = sm_census_l_0[l_idx];
        unsigned int l2_1 = sm_census_l_1[l_idx];

        unsigned int r1_1 = sm_census_r_0[l_idx2];
        unsigned int r2_1 = sm_census_r_1[l_idx2];

        for (int d = 0; d < num_disp; ++d)
        {
            int r_offset = tx + sm_padding_l + (d - zero_disp);
            int r_idx = r_offset + ty_sm_cols;
            int l_offset = tx + sm_padding_r - (d - zero_disp);
            int r_idx2 = l_offset + ty_sm_cols;

            unsigned int r1_2 = sm_census_r_0[r_idx];
            unsigned int r2_2 = sm_census_r_1[r_idx];
            
            unsigned int l1_2 = sm_census_l_0[r_idx2];
            unsigned int l2_2 = sm_census_l_1[r_idx2];
            
            
            float cost_hamming = (float) alu_hamdist_32(l1_1, r1_2);
            cost_hamming += (float) alu_hamdist_32(l2_1, r2_2);
            cost_l[d][gx + gy_num_cols] += cost_hamming * 0.33333333333f;
            
            cost_hamming = (float) alu_hamdist_32(r1_1, l1_2);
            cost_hamming += (float) alu_hamdist_32(r2_1, l2_2);
            cost_r[d][gx + gy_num_cols] += cost_hamming * 0.33333333333f;
        }
        __syncthreads();
    }
}

// Left & Right Calculation into 1 Module
__global__ void ci_census_kernel_4(unsigned long long *census_l, unsigned long long *census_r, 
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

        int l_idx = tx + sm_padding_r + ty_sm_cols;
        int l_idx2 = tx + sm_padding_l + ty_sm_cols;
        unsigned long long l1 = sm_census_l[l_idx];
        unsigned long long r1 = sm_census_r[l_idx2];

        for (int d = 0; d < num_disp; ++d)
        {
            int r_offset = tx + sm_padding_l + (d - zero_disp);
            int r_idx = r_offset + ty_sm_cols;
            int l_offset = tx + sm_padding_r - (d - zero_disp);
            int r_idx2 = l_offset + ty_sm_cols;

            unsigned long long r2 = sm_census_r[r_idx];
            unsigned long long l2 = sm_census_l[r_idx2];
            
            float cost_hamming = (float) alu_hamdist_64(l1, r2);
            cost_l[d][gx + gy_num_cols] += cost_hamming * 0.33333333333f;
            
            cost_hamming = (float) alu_hamdist_64(r1, l2);
            cost_r[d][gx + gy_num_cols] += cost_hamming * 0.33333333333f;
        }
        __syncthreads();
    }
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
