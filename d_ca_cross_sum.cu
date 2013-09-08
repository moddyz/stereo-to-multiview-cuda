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

    extern __shared__ float sm_cost_mem[];
    float* sm_cost = sm_cost_mem;
    
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
