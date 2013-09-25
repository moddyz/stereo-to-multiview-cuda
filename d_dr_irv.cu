#ifndef D_DR_IRV_KERNEL 
#define D_DR_IRV_KERNEL
#include "d_dr_irv.h"
#include "cuda_utils.h"
#include <math.h>

#define CROSS_ARM_COUNT 4

typedef enum
{
    CROSS_ARM_UP = 0,
    CROSS_ARM_DOWN,
    CROSS_ARM_LEFT,
    CROSS_ARM_RIGHT
} cross_arm_e;

__global__ void dr_irv_kernel_3(float *disp, unsigned char *outliers,
                                int *max_disp, int *reliable,
                                int thresh_s, float thresh_h,
                                int num_rows, int num_cols, 
                                int num_disp, int zero_disp)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int gx_gy_num_cols = gx + gy * num_cols;
    
    int total_reliable = reliable[gx_gy_num_cols];
    int max_d = max_disp[gx_gy_num_cols];

    if (outliers[gx_gy_num_cols] != 0)
    {
        if (total_reliable > thresh_s && (float)(max_d + zero_disp)/(float)total_reliable > thresh_h)
        {
            outliers[gx_gy_num_cols] = 0;
            reliable[gx_gy_num_cols] += 1;
            disp[gx_gy_num_cols] = max_d;
        }
    }
}

__global__ void dr_irv_pre_kernel(float *disp, unsigned char *outliers, unsigned char **cross,
                                   int *max_disp, int *reliable,
                                   int num_rows, int num_cols, 
                                   int num_disp, int zero_disp,
                                   int sm_width, int sm_height, int sm_sz, int sm_padding)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float sm_disp[];
    unsigned char *sm_outliers = (unsigned char*) (sm_disp + sm_sz);

    // Fill shared memory

    for (int gsy = gy - sm_padding, tsy = ty;
         tsy < sm_height;
         gsy += blockDim.y, tsy += blockDim.y)
    {
        for (int gsx = gx - sm_padding, tsx = tx;
             tsx < sm_width;
             gsx += blockDim.x, tsx += blockDim.x)
        {
            int sidx = tsx + tsy * sm_width;
            int gidx = min(max(gsx, 0), num_cols - 1) + min(max(gsy, 0), num_rows - 1) * num_cols;
            
            sm_outliers[sidx] = outliers[gidx];
            sm_disp[sidx] = disp[gidx];
        }
    }

    // Compute
    int gx_gy_num_cols = gx + gy * num_cols;
    
    int cross_u = cross[CROSS_ARM_UP][gx_gy_num_cols];
    int cross_d = cross[CROSS_ARM_DOWN][gx_gy_num_cols];

    unsigned char go = outliers[gx_gy_num_cols];

    int max_bin = 0;
    int max_d = sm_disp[tx + sm_padding + (ty + sm_padding) * sm_width];
    int total_reliable = 0;

    if (go != 0)
    {
        int dhist[65];
        for (int i = 0; i < 65; ++i)
            dhist[i] = 0;
        for (int y = -cross_u; y <= cross_d; ++y)
        {
            int cross_l = cross[CROSS_ARM_LEFT][gx + (gy + y) * num_cols];
            int cross_r = cross[CROSS_ARM_RIGHT][gx + (gy + y) * num_cols];
            for (int x = -cross_l; x <= cross_r; ++x)
            {
                int sidx = tx + sm_padding + x + (ty + sm_padding + y) * sm_width;
                unsigned char o = sm_outliers[sidx];

                if (o == 0)
                {
                    int d = (int) sm_disp[sidx];
                    dhist[d + zero_disp]++;
                    total_reliable++;
                }
            }
        }
        for (int i = 0; i < 65; ++i)
        {
            int curr_bin = dhist[i];
            
            if (max_bin < curr_bin)
            {
                max_bin = curr_bin;
                max_d = i - zero_disp;
            }
        }

        max_disp[gx_gy_num_cols] = max_d;
        reliable[gx_gy_num_cols] = total_reliable;
    }

}

__global__ void dr_irv_kernel_2(float *disp, unsigned char *outliers, unsigned char **cross,
                                int thresh_s, float thresh_h,
                                int num_rows, int num_cols, 
                                int num_disp, int zero_disp,
                                int sm_width, int sm_height, int sm_sz, int sm_padding)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float sm_disp[];
    unsigned char *sm_outliers = (unsigned char*) (sm_disp + sm_sz);

    // Fill shared memory

    for (int gsy = gy - sm_padding, tsy = ty;
         tsy < sm_height;
         gsy += blockDim.y, tsy += blockDim.y)
    {
        for (int gsx = gx - sm_padding, tsx = tx;
             tsx < sm_width;
             gsx += blockDim.x, tsx += blockDim.x)
        {
            int sidx = tsx + tsy * sm_width;
            int gidx = min(max(gsx, 0), num_cols - 1) + min(max(gsy, 0), num_rows - 1) * num_cols;
            
            sm_outliers[sidx] = outliers[gidx];
            sm_disp[sidx] = disp[gidx];
        }
    }

    // Compute
    int gx_gy_num_cols = gx + gy * num_cols;
    
    int cross_u = cross[CROSS_ARM_UP][gx_gy_num_cols];
    int cross_d = cross[CROSS_ARM_DOWN][gx_gy_num_cols];

    unsigned char go = outliers[gx_gy_num_cols];

    int max_bin = 0;
    int max_d = sm_disp[tx + sm_padding + (ty + sm_padding) * sm_width];
    int total_reliable = 0;

    if (go != 0)
    {
        int dhist[65];
        for (int i = 0; i < 65; ++i)
            dhist[i] = 0;
        for (int y = -cross_u; y <= cross_d; ++y)
        {
            int cross_l = cross[CROSS_ARM_LEFT][gx + (gy + y) * num_cols];
            int cross_r = cross[CROSS_ARM_RIGHT][gx + (gy + y) * num_cols];
            for (int x = -cross_l; x <= cross_r; ++x)
            {
                int sidx = tx + sm_padding + x + (ty + sm_padding + y) * sm_width;
                unsigned char o = sm_outliers[sidx];

                if (o == 0)
                {
                    int d = (int) sm_disp[sidx];
                    dhist[d + zero_disp]++;
                    total_reliable++;
                }
            }
        }
        for (int i = 0; i < 65; ++i)
        {
            int curr_bin = dhist[i];
            
            if (max_bin < curr_bin)
            {
                max_bin = curr_bin;
                max_d = i - zero_disp;
            }
        }
        if (total_reliable > thresh_s && (float)(max_d + zero_disp)/(float)total_reliable > thresh_h)
        {
            outliers[gx_gy_num_cols] = 0;
            disp[gx_gy_num_cols] = max_d;
        }
    }
}

__global__ void dr_irv_kernel(float *disp, unsigned char *outliers, unsigned char **cross,
                              int thresh_s, float thresh_h,
                              int num_rows, int num_cols, int num_disp, int zero_disp)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int gx_gy_num_cols = gx + gy * num_cols;
    
    int cross_u = cross[CROSS_ARM_UP][gx_gy_num_cols];
    int cross_d = cross[CROSS_ARM_DOWN][gx_gy_num_cols];

    unsigned char go = outliers[gx_gy_num_cols];

    int dhist[65];
    
    for (int i = 0; i < num_disp + 1; ++i)
        dhist[i] = 0;
    
    int max_bin = 0;
    int max_d = disp[gx_gy_num_cols];
    int total_reliable = 0;

    if (go != 0)
    {
        for (int y = -cross_u; y <= cross_d; ++y)
        {
            int cross_l = cross[CROSS_ARM_LEFT][gx + (gy + y) * num_cols];
            int cross_r = cross[CROSS_ARM_RIGHT][gx + (gy + y) * num_cols];
            for (int x = -cross_l; x <= cross_r; ++x)
            {
                int idx = gx + x + (gy + y) * num_cols;
                
                unsigned char o = outliers[idx];
                if (o == 0)
                {
                    int d = (int) disp[idx];
                    
                    dhist[d + zero_disp] += 1;
                    
                    int curr_bin = dhist[d + zero_disp];

                    if (max_bin < curr_bin)
                    {
                        max_bin = curr_bin;
                        max_d = d;
                    }

                    total_reliable++;
                }
            }
        }
    }

    if (total_reliable > thresh_s && (float)(max_d + zero_disp)/(float)total_reliable > thresh_h)
    {
        outliers[gx_gy_num_cols] = 0;
        disp[gx_gy_num_cols] = max_d;
    }
}

void dr_irv( float* disp, unsigned char* outliers, unsigned char **cross,
             int thresh_s, float thresh_h,
             int num_rows, int num_cols, int num_disp, int zero_disp,
             int usd,
             int iterations)
{
    cudaEventPair_t timer;
    
    /////////////////////// 
    // DEVICE PARAMETERS //
    ///////////////////////
    
    size_t bw = 32;
    size_t bh = 32;
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
    
    float* d_disp;

    checkCudaError(cudaMalloc(&d_disp, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMemcpy(d_disp, disp, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));

    unsigned char* d_outliers;

    checkCudaError(cudaMalloc(&d_outliers, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMemcpy(d_outliers, outliers, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyHostToDevice));
   
    unsigned char** d_cross;
    checkCudaError(cudaMalloc(&d_cross, sizeof(unsigned char*) * CROSS_ARM_COUNT));

    unsigned char** h_cross = (unsigned char**) malloc(sizeof(unsigned char*) * CROSS_ARM_COUNT);
    
    for (int i = 0; i < CROSS_ARM_COUNT; ++i)
    {
        checkCudaError(cudaMalloc(&h_cross[i], sizeof(unsigned char) * num_rows * num_cols));
        checkCudaError(cudaMemcpy(h_cross[i], cross[i], sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyHostToDevice));
    }

    checkCudaError(cudaMemcpy(d_cross, h_cross, sizeof(unsigned char*) * CROSS_ARM_COUNT, cudaMemcpyHostToDevice));

    int* d_max_disp;
    checkCudaError(cudaMalloc(&d_max_disp, sizeof(int) * num_rows * num_cols));
    
    int* d_reliable;
    checkCudaError(cudaMalloc(&d_reliable, sizeof(int) * num_rows * num_cols));
   
    int sw = bw + 2 * usd;
    int sh = bh + 2 * usd;
    int sm_sz = sw * sh;
    
    startCudaTimer(&timer);
    dr_irv_pre_kernel<<<grid_sz, block_sz, sizeof(float) * sm_sz + sizeof(unsigned char) * sm_sz>>>(d_disp, d_outliers, d_cross, d_max_disp, d_reliable, num_rows, num_cols, num_disp, zero_disp, sw, sh, sm_sz, usd);
    stopCudaTimer(&timer, "Disparity Refinement Iterative Region Voting Preprocessor Kernel");
    
    for (int i = 0; i < iterations; ++i)
    {
        startCudaTimer(&timer);
        dr_irv_kernel_3<<<grid_sz, block_sz>>>(d_disp, d_outliers, d_max_disp, d_reliable, thresh_s, thresh_h, num_rows, num_cols, num_disp, zero_disp); 
        stopCudaTimer(&timer, "Disparity Refinement Iterative Region Voting Kernel");
    }

    checkCudaError(cudaMemcpy(disp, d_disp, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));

    cudaFree(d_disp);
    cudaFree(d_outliers);
    cudaFree(d_cross);
    free(h_cross);
}

#endif
