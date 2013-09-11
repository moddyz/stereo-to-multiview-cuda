#ifndef D_CA_CROSS_SUM_KERNEL 
#define D_CA_CROSS_SUM_KERNEL
#include "d_ca_cross_sum.h"
#include "cuda_utils.h"
#include <math.h>

typedef enum
{
    CROSS_ARM_UP = 0,
    CROSS_ARM_DOWN,
    CROSS_ARM_LEFT,
    CROSS_ARM_RIGHT
} cross_arm_e;


__global__ void cost_copy_kernel(float **cost, float** cost_t, 
                                 int num_disp, int num_rows, int num_cols)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    for (int d = 0; d < num_disp; ++d)
        cost_t[d][gx + gy * num_cols] = cost[d][gx + gy * num_cols];
}

__global__ void cost_transpose_kernel_4(float **cost, float** cost_t, 
                                        int num_disp, int num_rows, int num_cols,
                                        int tile_width, int tile_height)
{
    __shared__ float sm_cost[32][32 + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int width = num_cols;
    int height = num_rows;

    for (int d = 0; d < num_disp; ++d)
    {
        int x = blockIdx.x * tile_width + threadIdx.x;
        int y = blockIdx.y * tile_height + threadIdx.y;
        
        for (int j = 0; j < tile_height; j += blockDim.y)
            sm_cost[ty + j][tx] = cost[d][(y + j) * width + x];

        __syncthreads();
        
        x = blockIdx.y * tile_height + threadIdx.x;
        y = blockIdx.x * tile_width + threadIdx.y;
        
        for (int j = 0; j < tile_width; j += blockDim.y)
                cost_t[d][(y + j) * height + x] = sm_cost[tx][ty + j];
        __syncthreads();
    }
}

__global__ void cost_transpose_kernel_3(float **cost, float** cost_t, 
                                        int num_disp, int num_rows, int num_cols,
                                        int tile_width, int tile_height)
{
    extern __shared__ float sm_cost[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int width = num_cols;
    int height = num_rows;

    for (int d = 0; d < num_disp; ++d)
    {
        int x = blockIdx.x * tile_width + threadIdx.x;
        int y = blockIdx.y * tile_height + threadIdx.y;
        
        for (int j = 0; j < tile_height; j += blockDim.y)
            sm_cost[tx * tile_height + ty + j] = cost[d][(y + j) * width + x];

        __syncthreads();
        
        x = blockIdx.y * tile_height + threadIdx.x;
        y = blockIdx.x * tile_width + threadIdx.y;
        
        for (int k = 0; k < tile_height; k += blockDim.x)
        {
            if (tx + k < tile_height)
            {
                for (int j = 0; j < tile_width; j += blockDim.y)
                    if (ty + j < tile_width) 
                        cost_t[d][(y + j) * height + x + k] = sm_cost[tx + k + (ty + j) * tile_height];
            }
            else
                break;
        }
        __syncthreads();
    }
}


__global__ void cost_transpose_kernel_2(float **cost, float** cost_t, 
                                        int num_disp, int num_rows, int num_cols,
                                        int tile_width, int tile_height)
{
    extern __shared__ float sm_cost[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int width = num_cols;
    int height = num_rows;

    for (int d = 0; d < num_disp; ++d)
    {
        int x = blockIdx.x * tile_width + threadIdx.x;
        int y = blockIdx.y * tile_height + threadIdx.y;
        
        for (int j = 0; j < tile_height; j += blockDim.y)
            sm_cost[tx * tile_height + ty + j] = cost[d][(y + j) * width + x];

        __syncthreads();
        
        x = blockIdx.y * tile_height + threadIdx.x;
        y = blockIdx.x * tile_width + threadIdx.y;
        
        if (threadIdx.x < tile_height)
        {
            for (int j = 0; j < tile_width; j += blockDim.y)
                cost_t[d][(y + j) * height + x] = sm_cost[tx + (ty + j) * tile_height];
        }
        __syncthreads();
    }
}

__global__ void cost_transpose_kernel(float **cost, float** cost_t, 
                                      int num_disp, int num_rows, int num_cols)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;
    
    for (int d = 0; d < num_disp; ++d)
        cost_t[d][gy + gx * num_rows] = cost[d][gx + gy * num_cols];
}

__global__ void ca_cross_vhsum_kernel_2(float** cost, float** acost, unsigned char** cross,
                                       int num_disp, int num_rows, int num_cols,
                                       int sm_cols, int sm_sz, int sm_padding, 
                                       int ipt)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float sm_mem[];
    float *sm_cost = sm_mem;
    int *sm_arm_l = (int*) sm_cost + sm_cols;
    int *sm_arm_r = sm_arm_l + blockDim.x * ipt;
    
    int ty_sm_cols = ty * sm_cols;
    int gy_num_cols = gy * num_cols;
    
    for (int i = 0, x = tx; i < ipt; ++i, x += blockDim.x)
    {
        sm_arm_l[x] = (int) cross[CROSS_ARM_UP][gy + (gx + i * blockDim.x) * num_rows];
        sm_arm_r[x] = (int) cross[CROSS_ARM_DOWN][gy + (gx + i * blockDim.x) * num_rows];
    }

    for (int d = 0; d < num_disp; ++d)
    {
        for (int gsx = gx - sm_padding, tsx = tx; tsx < sm_cols; gsx += blockDim.x, tsx += blockDim.x)
        {
            int sm_idx = tsx + ty_sm_cols;
            int gm_idx = min(max(gsx, 0), num_cols - 1) + gy_num_cols;

            sm_cost[sm_idx] = cost[d][gm_idx];
        }
        __syncthreads();
         
        for (int i = 0, x = tx; i < ipt; ++i, x += blockDim.x)
        {
            float asum = 0;
            int bdi = blockDim.x * i;
            for (int ax = tx + bdi - sm_arm_l[x] + sm_padding; ax < tx + bdi + sm_arm_r[x] + sm_padding; ++ax)
                asum = asum + sm_cost[ax + ty_sm_cols];
            
            acost[d][gx + bdi + gy_num_cols] = asum;
        }
        __syncthreads();
    }
}

__global__ void ca_cross_vhsum_kernel(float** cost, float** acost, unsigned char** cross,
                                       int num_disp, int num_rows, int num_cols,
                                       int sm_cols, int sm_sz, int sm_padding)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float sm_cost_mem[];
    float* sm_cost = sm_cost_mem;
    
    int ty_sm_cols = ty * sm_cols;
    int gy_num_cols = gy * num_cols;

    int arm_l = (int) cross[CROSS_ARM_UP][gy + gx * num_rows];
    int arm_r = (int) cross[CROSS_ARM_DOWN][gy + gx * num_rows];
    
    for (int d = 0; d < num_disp; ++d)
    {
        for (int gsx = gx - sm_padding, tsx = tx; tsx < sm_cols; gsx += blockDim.x, tsx += blockDim.x)
        {
            int sm_idx = tsx + ty_sm_cols;
            int gm_idx = min(max(gsx, 0), num_cols - 1) + gy_num_cols;

            sm_cost[sm_idx] = cost[d][gm_idx];
        }
        __syncthreads();

        float asum = 0;
        for (int ax = tx - arm_l + sm_padding; ax < tx + arm_r + sm_padding; ++ax)
        {
            asum = asum + sm_cost[ax + ty_sm_cols];
        }
        acost[d][gx + gy_num_cols] = asum;
        __syncthreads();
    }
}

__global__ void ca_cross_hsum_kernel_3(float** cost, float** acost, unsigned char** cross,
                                       int num_disp, int num_rows, int num_cols,
                                       int sm_cols, int sm_sz, int sm_padding, 
                                       int ipt)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float sm_mem[];
    float *sm_cost = sm_mem;
    int *sm_arm_l = (int*) sm_cost + sm_cols;
    int *sm_arm_r = sm_arm_l + blockDim.x * ipt;
    
    int ty_sm_cols = ty * sm_cols;
    int gy_num_cols = gy * num_cols;
    
    for (int i = 0, x = tx; i < ipt; ++i, x += blockDim.x)
    {
        sm_arm_l[x] = (int) cross[CROSS_ARM_LEFT][gx + i * blockDim.x + gy_num_cols];
        sm_arm_r[x] = (int) cross[CROSS_ARM_RIGHT][gx + i * blockDim.x + gy_num_cols];
    }

    for (int d = 0; d < num_disp; ++d)
    {
        for (int gsx = gx - sm_padding, tsx = tx; tsx < sm_cols; gsx += blockDim.x, tsx += blockDim.x)
        {
            int sm_idx = tsx + ty_sm_cols;
            int gm_idx = min(max(gsx, 0), num_cols - 1) + gy_num_cols;

            sm_cost[sm_idx] = cost[d][gm_idx];
        }
        __syncthreads();
         
        for (int i = 0, x = tx; i < ipt; ++i, x += blockDim.x)
        {
            float asum = 0;
            int bdi = blockDim.x * i;
            for (int ax = tx + bdi - sm_arm_l[x] + sm_padding; ax < tx + bdi + sm_arm_r[x] + sm_padding; ++ax)
                asum = asum + sm_cost[ax + ty_sm_cols];
            
            acost[d][gx + bdi + gy_num_cols] = asum;
        }
        __syncthreads();
    }
}

__global__ void ca_cross_hsum_kernel_2(float** cost, float** acost, unsigned char** cross,
                                       int num_disp, int num_rows, int num_cols,
                                       int sm_cols, int sm_sz, int sm_padding)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float sm_cost[];
    
    int ty_sm_cols = ty * sm_cols;
    int gy_num_cols = gy * num_cols;

    int arm_l = (int) cross[CROSS_ARM_LEFT][gx + gy_num_cols];
    int arm_r = (int) cross[CROSS_ARM_RIGHT][gx + gy_num_cols];
    
    for (int d = 0; d < num_disp; ++d)
    {
        for (int gsx = gx - sm_padding, tsx = tx; tsx < sm_cols; gsx += blockDim.x, tsx += blockDim.x)
        {
            int sm_idx = tsx + ty_sm_cols;
            int gm_idx = min(max(gsx, 0), num_cols - 1) + gy_num_cols;

            sm_cost[sm_idx] = cost[d][gm_idx];
        }
        __syncthreads();

        float asum = 0;
        for (int ax = tx - arm_l + sm_padding; ax < tx + arm_r + sm_padding; ++ax)
        {
            asum = asum + sm_cost[ax + ty_sm_cols];
        }
        acost[d][gx + gy_num_cols] = asum;
        __syncthreads();
    }
}

__global__ void ca_cross_hsum_kernel(float** cost, float** acost, unsigned char** cross,
                                     int num_disp, int num_rows, int num_cols)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int arm_l = (int) cross[CROSS_ARM_LEFT][gx + gy * num_cols];
    int arm_r = (int) cross[CROSS_ARM_RIGHT][gx + gy * num_cols];
    for (int d = 0; d < num_disp; ++d)
    {
        float asum = 0;
        for (int ax = gx - arm_l; ax < gx + arm_r; ++ax)
        {
            asum = asum + cost[d][ax + gy * num_cols];
        }
        acost[d][gx + gy * num_cols] = asum;
    }
}

__global__ void ca_cross_vsum_kernel_2(float** cost, float** acost, unsigned char** cross,
                                       int num_disp, int num_rows, int num_cols,
                                       int sm_rows, int sm_sz, int sm_padding)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float sm_cost_mem[];
    float* sm_cost = sm_cost_mem;

    int gy_num_cols = gy * num_cols;
    int tx_num_rows = tx * num_rows;

    int arm_u = (int) cross[CROSS_ARM_UP][gx + gy_num_cols];
    int arm_d = (int) cross[CROSS_ARM_DOWN][gx + gy_num_cols];
    for (int d = 0; d < num_disp; ++d)
    {
        for (int gsy = gy - sm_padding, tsy = ty; tsy < sm_rows; gsy += blockDim.y, tsy += blockDim.y)
        {
            int sm_idx = tsy + tx_num_rows;
            int gm_idx = min(max(gsy, 0), num_rows - 1) * num_cols + gx;

            sm_cost[sm_idx] = cost[d][gm_idx];
        }

        __syncthreads();

        float asum = 0;
        for (int ay = ty + sm_padding - arm_u; ay < ty + sm_padding + arm_d; ++ay)
        {
            asum = asum + sm_cost[ay + tx_num_rows];
        }
        
        acost[d][gx + gy_num_cols] = asum;

        __syncthreads();
    }
}

__global__ void ca_cross_vsum_kernel(float** cost, float** acost, unsigned char** cross,
                                     int num_disp, int num_rows, int num_cols)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int arm_u = (int) cross[CROSS_ARM_UP][gx + gy * num_cols];
    int arm_d = (int) cross[CROSS_ARM_DOWN][gx + gy * num_cols];
    for (int d = 0; d < num_disp; ++d)
    {
        float asum = 0;
        for (int ay = gy - arm_u; ay < gy + arm_d; ++ay)
        {
            asum = asum + cost[d][gx + ay * num_cols];
        }
        acost[d][gx + gy * num_cols] = asum;
    }
}

#endif
