#ifndef D_CA_CROSS_KERNEL 
#define D_CA_CROSS_KERNEL
#include "d_ca_cross.h"
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

__global__ void ca_cross_construction_kernel(unsigned char* img, unsigned char** cross,
                                             float ucd, float lcd, int usd, int lsd,
                                             int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;

    unsigned char a_color_b = img[(tx + ty * num_cols) * elem_sz];
    unsigned char a_color_g = img[(tx + ty * num_cols) * elem_sz + 1];
    unsigned char a_color_r = img[(tx + ty * num_cols) * elem_sz + 2];
    
    // Upper arm
    for (int y = 1; y <= usd; ++y)
    {
        if (ty - y < 0)
            break;
        
        cross[CROSS_ARM_UP][tx + ty * num_cols] = y;
        
        int c_color_b = (int) img[(tx + (ty - y) * num_cols) * elem_sz];
        int c_color_g = (int) img[(tx + (ty - y) * num_cols) * elem_sz + 1];
        int c_color_r = (int) img[(tx + (ty - y) * num_cols) * elem_sz + 2];
        
        int p_color_b = (int) img[(tx + (ty - y) * num_cols) * elem_sz];
        int p_color_g = (int) img[(tx + (ty - y) * num_cols) * elem_sz + 1];
        int p_color_r = (int) img[(tx + (ty - y) * num_cols) * elem_sz + 2];

        int ac_mad = max(max(abs(c_color_b - a_color_b), abs(c_color_g - a_color_g)), abs(c_color_r - a_color_r));
        int cp_mad = max(max(abs(c_color_b - p_color_b), abs(c_color_g - p_color_g)), abs(c_color_r - p_color_r));

        if (y > lsd)
        {
            if ((float) ac_mad > ucd)
                break;
        }
        else
        {
            if ((float) ac_mad > lcd || (float) cp_mad > lcd)
                break;
        }
    }

    // Down arm
    for (int y = 1; y <= usd; ++y)
    {
        if (ty + y < 0)
            break;
        
        cross[CROSS_ARM_DOWN][tx + ty * num_cols] = y;
        
        int c_color_b = (int) img[(tx + (ty + y) * num_cols) * elem_sz];
        int c_color_g = (int) img[(tx + (ty + y) * num_cols) * elem_sz + 1];
        int c_color_r = (int) img[(tx + (ty + y) * num_cols) * elem_sz + 2];
        
        int p_color_b = (int) img[(tx + (ty + y) * num_cols) * elem_sz];
        int p_color_g = (int) img[(tx + (ty + y) * num_cols) * elem_sz + 1];
        int p_color_r = (int) img[(tx + (ty + y) * num_cols) * elem_sz + 2];

        int ac_mad = max(max(abs(c_color_b - a_color_b), abs(c_color_g - a_color_g)), abs(c_color_r - a_color_r));
        int cp_mad = max(max(abs(c_color_b - p_color_b), abs(c_color_g - p_color_g)), abs(c_color_r - p_color_r));

        if (y > lsd)
        {
            if ((float) ac_mad > ucd)
                break;
        }
        else
        {
            if ((float) ac_mad > lcd || (float) cp_mad > lcd)
                break;
        }
    }
    
    // Left arm
    for (int x = 1; x <= usd; ++x)
    {
        if (tx - x < 0)
            break;
        
        cross[CROSS_ARM_LEFT][tx + ty * num_cols] = x;
        
        int c_color_b = (int) img[(tx - x + ty * num_cols) * elem_sz];
        int c_color_g = (int) img[(tx - x + ty * num_cols) * elem_sz + 1];
        int c_color_r = (int) img[(tx - x + ty * num_cols) * elem_sz + 2];
        
        int p_color_b = (int) img[(tx - x + ty * num_cols) * elem_sz];
        int p_color_g = (int) img[(tx - x + ty * num_cols) * elem_sz + 1];
        int p_color_r = (int) img[(tx - x + ty * num_cols) * elem_sz + 2];

        int ac_mad = max(max(abs(c_color_b - a_color_b), abs(c_color_g - a_color_g)), abs(c_color_r - a_color_r));
        int cp_mad = max(max(abs(c_color_b - p_color_b), abs(c_color_g - p_color_g)), abs(c_color_r - p_color_r));

        if (x > lsd)
        {
            if ((float) ac_mad > ucd)
                break;
        }
        else
        {
            if ((float) ac_mad > lcd || (float) cp_mad > lcd)
                break;
        }
    }
    
    // Right arm
    for (int x = 1; x <= usd; ++x)
    {
        if (tx + x < 0)
            break;
        
        cross[CROSS_ARM_RIGHT][tx + ty * num_cols] = x;
        
        int c_color_b = (int) img[(tx + x + ty * num_cols) * elem_sz];
        int c_color_g = (int) img[(tx + x + ty * num_cols) * elem_sz + 1];
        int c_color_r = (int) img[(tx + x + ty * num_cols) * elem_sz + 2];
        
        int p_color_b = (int) img[(tx + x + ty * num_cols) * elem_sz];
        int p_color_g = (int) img[(tx + x + ty * num_cols) * elem_sz + 1];
        int p_color_r = (int) img[(tx + x + ty * num_cols) * elem_sz + 2];

        int ac_mad = max(max(abs(c_color_b - a_color_b), abs(c_color_g - a_color_g)), abs(c_color_r - a_color_r));
        int cp_mad = max(max(abs(c_color_b - p_color_b), abs(c_color_g - p_color_g)), abs(c_color_r - p_color_r));

        if (x > lsd)
        {
            if ((float) ac_mad > ucd)
                break;
        }
        else
        {
            if ((float) ac_mad > lcd || (float) cp_mad > lcd)
                break;
        }
    }
}

void ca_cross(unsigned char* img_l, unsigned char* img_r, float** cost_l, float** cost_r,
              float** acost_l, float** acost_r, float ucd, float lcd, int usd, int lsd,
              int num_rows, int num_cols, int elem_sz)
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
    
    //////////////////////// 
    // CROSS CONSTRUCTION //
    ////////////////////////

    unsigned char* d_img_l;
    unsigned char* d_img_r;

    checkCudaError(cudaMalloc(&d_img_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMalloc(&d_img_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz));

    checkCudaError(cudaMemcpy(d_img_l, img_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_img_r, img_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));
   
    unsigned char** d_cross_l;
    unsigned char** d_cross_r;
    checkCudaError(cudaMalloc(&d_cross_l, sizeof(unsigned char*) * CROSS_ARM_COUNT));
    checkCudaError(cudaMalloc(&d_cross_r, sizeof(unsigned char*) * CROSS_ARM_COUNT));

    unsigned char** h_cross_l = (unsigned char**) malloc(sizeof(unsigned char*) * CROSS_ARM_COUNT);
    unsigned char** h_cross_r = (unsigned char**) malloc(sizeof(unsigned char*) * CROSS_ARM_COUNT);
    
    for (int i = 0; i < CROSS_ARM_COUNT; ++i)
    {
        checkCudaError(cudaMalloc(&h_cross_l[i], sizeof(unsigned char) * num_rows * num_cols));
        checkCudaError(cudaMalloc(&h_cross_r[i], sizeof(unsigned char) * num_rows * num_cols));
    }

    checkCudaError(cudaMemcpy(d_cross_l, h_cross_l, sizeof(unsigned char*) * CROSS_ARM_COUNT, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_cross_r, h_cross_r, sizeof(unsigned char*) * CROSS_ARM_COUNT, cudaMemcpyHostToDevice));
    
    // Launch kernel
    startCudaTimer(&timer);
    ca_cross_construction_kernel<<<grid_sz, block_sz>>>(d_img_l, d_cross_l, ucd, lcd, usd, lsd, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Cross Aggragation - Cross Construciton Kernel");
    
    startCudaTimer(&timer);
    ca_cross_construction_kernel<<<grid_sz, block_sz>>>(d_img_r, d_cross_r, ucd, lcd, usd, lsd, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Cross Aggragation - Cross Construciton Kernel");

    ///////////////////
    // DE-ALLOCATION // 
    ///////////////////

    cudaFree(d_img_l);
    cudaFree(d_img_r);
    cudaFree(d_cross_l);
    cudaFree(d_cross_r);
    for (int i = 0; i < CROSS_ARM_COUNT; ++i)
    {
        cudaFree(h_cross_l[i]);
        cudaFree(h_cross_r[i]);
    }
    free(h_cross_l);
    free(h_cross_r);
}

#endif
