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
    
    cross[CROSS_ARM_UP][tx + ty * num_cols] = 0;
    cross[CROSS_ARM_DOWN][tx + ty * num_cols] = 0;
    cross[CROSS_ARM_LEFT][tx + ty * num_cols] = 0;
    cross[CROSS_ARM_RIGHT][tx + ty * num_cols] = 0;
        
    int p_color_b = a_color_b;
    int p_color_g = a_color_g;
    int p_color_r = a_color_r;
    
    // Upper arm
    for (int y = 1; y <= usd; ++y)
    {
        if (ty - y < 0)
            break;
        
        cross[CROSS_ARM_UP][tx + ty * num_cols] = (unsigned char) y;
        
        int c_color_b = (int) img[(tx + (ty - y) * num_cols) * elem_sz];
        int c_color_g = (int) img[(tx + (ty - y) * num_cols) * elem_sz + 1];
        int c_color_r = (int) img[(tx + (ty - y) * num_cols) * elem_sz + 2];

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

        p_color_b = c_color_b;
        p_color_g = c_color_g;
        p_color_r = c_color_r;
    }
        
    p_color_b = a_color_b;
    p_color_g = a_color_g;
    p_color_r = a_color_r;

    // Down arm
    for (int y = 1; y <= usd; ++y)
    {
        if ((ty + y) > (num_rows - 1))
            break;
        
        cross[CROSS_ARM_DOWN][tx + ty * num_cols] = (unsigned char) y;
        
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
        p_color_b = c_color_b;
        p_color_g = c_color_g;
        p_color_r = c_color_r;
    }
        
    p_color_b = a_color_b;
    p_color_g = a_color_g;
    p_color_r = a_color_r;
    
    // Left arm
    for (int x = 1; x <= usd; ++x)
    {
        if (tx - x < 0)
            break;
        
        cross[CROSS_ARM_LEFT][tx + ty * num_cols] = (unsigned char) x;
        
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
        p_color_b = c_color_b;
        p_color_g = c_color_g;
        p_color_r = c_color_r;
    }
        
    p_color_b = a_color_b;
    p_color_g = a_color_g;
    p_color_r = a_color_r;
    
    // Right arm
    for (int x = 1; x <= usd; ++x)
    {
        if ((tx + x) > (num_cols - 1))
            break;
        
        cross[CROSS_ARM_RIGHT][tx + ty * num_cols] = (unsigned char) x;
        
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
        p_color_b = c_color_b;
        p_color_g = c_color_g;
        p_color_r = c_color_r;
    }
        
    p_color_b = a_color_b;
    p_color_g = a_color_g;
    p_color_r = a_color_r;
}

void d_ca_cross(unsigned char* d_img, float** d_cost,  
                float** d_acost, float** h_acost, float *d_acost_memory,
                float ucd, float lcd, int usd, int lsd,
                int num_disp, int num_rows, int num_cols, int elem_sz)
{
    //////////////// 
    // PARAMETERS //
    ////////////////

    size_t img_sz = num_rows * num_cols;
    
    //////////////////////// 
    // CROSS CONSTRUCTION //
    ////////////////////////
    
    size_t bw = num_cols;
    size_t bh = 1;
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
    

    unsigned char** d_cross;
    checkCudaError(cudaMalloc(&d_cross, sizeof(unsigned char*) * CROSS_ARM_COUNT));

    unsigned char** h_cross = (unsigned char**) malloc(sizeof(unsigned char*) * CROSS_ARM_COUNT);

    unsigned char* d_cross_memory;
    checkCudaError(cudaMalloc(&d_cross_memory, sizeof(unsigned char) * img_sz * CROSS_ARM_COUNT));
    
    for (int i = 0; i < CROSS_ARM_COUNT; ++i)
        h_cross[i] = d_cross_memory + (i * img_sz);

    checkCudaError(cudaMemcpy(d_cross, h_cross, sizeof(unsigned char*) * CROSS_ARM_COUNT, cudaMemcpyHostToDevice));
    
    ca_cross_construction_kernel<<<grid_sz, block_sz>>>(d_img, d_cross, ucd, lcd, usd, lsd, num_rows, num_cols, elem_sz);
    cudaDeviceSynchronize();
    
    ///////////////////////////
    // CROSS-AGGRAGATE COSTS // 
    ///////////////////////////
	int sm_cols = bw;
	int sm_sz = sm_cols * bh;
	int sm_padding = 0;
	
    for (int d = 0; d < num_disp; ++d)
        h_acost[d] = d_acost_memory + (d * img_sz);

    checkCudaError(cudaMemcpy(d_acost, h_acost, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
   	
	size_t bw_t = 32;
    size_t bh_t = 8;
    size_t gw_t = (num_cols + bw_t - 1) / bw_t;
    size_t gh_t = (num_rows + bh_t - 1) / bh_t / 4;
    const dim3 block_sz_t(bw_t, bh_t, 1);
    const dim3 grid_sz_t(gw_t, gh_t, 1);
    
	size_t bw_t_v = 32;
    size_t bh_t_v = 8;
    size_t gw_t_v = (num_rows + bw_t_v - 1) / bw_t_v;
    size_t gh_t_v = (num_cols + bh_t_v - 1) / bh_t_v / 4;
    const dim3 block_sz_t_v(bw_t_v, bh_t_v, 1);
    const dim3 grid_sz_t_v(gw_t_v, gh_t_v, 1);

    int sm_width = 32;
    
    int ipt_s = 2;
    size_t bw_s = num_cols / ipt_s;
    size_t bh_s = 1;
    size_t gw_s = (num_cols + bw_s - 1) / bw_s / ipt_s;
    size_t gh_s = (num_rows + bh_s - 1) / bh_s;
    const dim3 block_sz_s(bw_s, bh_s, 1);
    const dim3 grid_sz_s(gw_s, gh_s, 1);
	
    int sm_cols_s = bw_s * ipt_s;
    int sm_arm_s = 2 * ipt_s * bw_s;
	int sm_sz_s = sm_cols_s + sm_arm_s +  1;
	int sm_padding_s = 0;
    
    int ipt_s_v = 2;
    size_t bw_s_v = num_rows / ipt_s_v;
    size_t bh_s_v = 1;
    size_t gw_s_v = (num_rows + bw_s_v - 1) / bw_s_v / ipt_s_v;
    size_t gh_s_v = (num_cols + bh_s_v - 1) / bh_s_v;
    const dim3 block_sz_s_v(bw_s_v, bh_s_v, 1);
    const dim3 grid_sz_s_v(gw_s_v, gh_s_v, 1);
	
    int sm_cols_s_v = bw_s_v * ipt_s_v;
    int sm_arm_s_v = 2 * ipt_s_v * bw_s_v;
	int sm_sz_s_v = sm_cols_s_v + sm_arm_s_v +  1;
	int sm_padding_s_v = 0;

	
    ca_cross_hsum_kernel_3<<<grid_sz_s, block_sz_s, sizeof(float) * sm_sz_s>>>(d_cost, d_acost, d_cross, num_disp, num_rows, num_cols, sm_cols_s, sm_sz_s, sm_padding_s, ipt_s); 
	cudaDeviceSynchronize();	

    cost_transpose_kernel_4<<<grid_sz_t, block_sz_t>>>(d_acost, d_cost, num_disp, num_rows, num_cols, sm_width, sm_width); 
	cudaDeviceSynchronize();	
	
    ca_cross_vhsum_kernel_2<<<grid_sz_s_v, block_sz_s_v, sizeof(float) * sm_sz_s_v>>>(d_cost, d_acost, d_cross, num_disp, num_cols, num_rows, sm_cols_s_v, sm_sz_s_v, sm_padding_s_v, ipt_s_v); 
	cudaDeviceSynchronize();	
	
    ca_cross_vhsum_kernel_2<<<grid_sz_s_v, block_sz_s_v, sizeof(float) * sm_sz_s_v>>>(d_acost, d_cost, d_cross, num_disp, num_cols, num_rows, sm_cols_s_v, sm_sz_s_v, sm_padding_s_v, ipt_s_v); 
	cudaDeviceSynchronize();	
	
    cost_transpose_kernel_4<<<grid_sz_t_v, block_sz_t_v>>>(d_cost, d_acost, num_disp, num_cols, num_rows, sm_width, sm_width); 
	cudaDeviceSynchronize();	
	
    ca_cross_hsum_kernel_3<<<grid_sz_s, block_sz_s, sizeof(float) * sm_sz_s>>>(d_acost, d_cost, d_cross, num_disp, num_rows, num_cols, sm_cols_s, sm_sz_s, sm_padding_s, ipt_s); 
	cudaDeviceSynchronize();	
    
    
    ///////////////////
    // DE-ALLOCATION // 
    ///////////////////
    
    cudaFree(d_cross_memory);
    cudaFree(d_cross);
    free(h_cross);
}

void ca_cross(unsigned char* img, float** cost, float** acost,
              float ucd, float lcd, int usd, int lsd,
              int num_disp, int num_rows, int num_cols, int elem_sz)
{
    cudaEventPair_t timer;
    
    /////////////////////// 
    // DEVICE PARAMETERS //
    ///////////////////////
    
    size_t bw = num_cols;
    size_t bh = 1;
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);

    
    //////////////////////// 
    // CROSS CONSTRUCTION //
    ////////////////////////

    unsigned char* d_img;

    checkCudaError(cudaMalloc(&d_img, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMemcpy(d_img, img, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));
   
    unsigned char** d_cross;
    checkCudaError(cudaMalloc(&d_cross, sizeof(unsigned char*) * CROSS_ARM_COUNT));

    unsigned char** h_cross = (unsigned char**) malloc(sizeof(unsigned char*) * CROSS_ARM_COUNT);
    
    for (int i = 0; i < CROSS_ARM_COUNT; ++i)
    {
        checkCudaError(cudaMalloc(&h_cross[i], sizeof(unsigned char) * num_rows * num_cols));
    }

    checkCudaError(cudaMemcpy(d_cross, h_cross, sizeof(unsigned char*) * CROSS_ARM_COUNT, cudaMemcpyHostToDevice));
    
    // Launch kernel
    startCudaTimer(&timer);
    ca_cross_construction_kernel<<<grid_sz, block_sz>>>(d_img, d_cross, ucd, lcd, usd, lsd, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Cross Aggragation - Cross Construciton Kernel");
    
    ///////////////////////////
    // CROSS-AGGRAGATE COSTS // 
    ///////////////////////////
    float** d_cost;

    checkCudaError(cudaMalloc(&d_cost, sizeof(float*) * num_disp));

    float** h_cost = (float**) malloc(sizeof(float*) * num_disp);
    
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMalloc(&h_cost[d], sizeof(float) * num_rows * num_cols));
        checkCudaError(cudaMemcpy(h_cost[d], cost[d], sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));
    }

    checkCudaError(cudaMemcpy(d_cost, h_cost, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    
    float** d_acost;
    checkCudaError(cudaMalloc(&d_acost, sizeof(float*) * num_disp));

    float** h_acost = (float**) malloc(sizeof(float*) * num_disp);
    
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMalloc(&h_acost[d], sizeof(float) * num_rows * num_cols));
    }

    checkCudaError(cudaMemcpy(d_acost, h_acost, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    
    int sm_cols = bw + 2 * usd;
	int sm_sz = sm_cols * bh;
	int sm_padding = usd;
    
	size_t bw_t = 32;
    size_t bh_t = 8;
    size_t gw_t = (num_cols + bw_t - 1) / bw_t;
    size_t gh_t = (num_rows + bh_t - 1) / bh_t / 4;
    const dim3 block_sz_t(bw_t, bh_t, 1);
    const dim3 grid_sz_t(gw_t, gh_t, 1);
    
	size_t bw_t_v = 32;
    size_t bh_t_v = 8;
    size_t gw_t_v = (num_rows + bw_t_v - 1) / bw_t_v;
    size_t gh_t_v = (num_cols + bh_t_v - 1) / bh_t_v / 4;
    const dim3 block_sz_t_v(bw_t_v, bh_t_v, 1);
    const dim3 grid_sz_t_v(gw_t_v, gh_t_v, 1);
    
    int sm_width= 32;
    
    int ipt_s = 2;
    size_t bw_s = num_cols / ipt_s;
    size_t bh_s = 1;
    size_t gw_s = (num_cols + bw_s - 1) / bw_s / ipt_s;
    size_t gh_s = (num_rows + bh_s - 1) / bh_s;
    const dim3 block_sz_s(bw_s, bh_s, 1);
    const dim3 grid_sz_s(gw_s, gh_s, 1);
	
    int sm_cols_s = bw_s * ipt_s;
    int sm_arm_s = 2 * ipt_s * bw_s;
	int sm_sz_s = sm_cols_s + sm_arm_s +  1;
	int sm_padding_s = 0;
    
    int ipt_s_v = 2;
    size_t bw_s_v = num_rows / ipt_s_v;
    size_t bh_s_v = 1;
    size_t gw_s_v = (num_rows + bw_s_v - 1) / bw_s_v / ipt_s_v;
    size_t gh_s_v = (num_cols + bh_s_v - 1) / bh_s_v;
    const dim3 block_sz_s_v(bw_s_v, bh_s_v, 1);
    const dim3 grid_sz_s_v(gw_s_v, gh_s_v, 1);
	
    int sm_cols_s_v = bw_s_v * ipt_s_v;
    int sm_arm_s_v = 2 * ipt_s_v * bw_s_v;
	int sm_sz_s_v = sm_cols_s_v + sm_arm_s_v +  1;
	int sm_padding_s_v = 0;

	
	startCudaTimer(&timer);
    ca_cross_hsum_kernel_3<<<grid_sz_s, block_sz_s, sizeof(float) * sm_sz_s>>>(d_cost, d_acost, d_cross, num_disp, num_rows, num_cols, sm_cols_s, sm_sz_s, sm_padding_s, ipt_s); 
    stopCudaTimer(&timer, "Cross Horizontal Sum #3");

	startCudaTimer(&timer);
    cost_transpose_kernel_4<<<grid_sz_t, block_sz_t>>>(d_acost, d_cost, num_disp, num_rows, num_cols, sm_width, sm_width); 
    stopCudaTimer(&timer, "Cost Transpose Kernel #4");
	
	startCudaTimer(&timer);
    ca_cross_vhsum_kernel_2<<<grid_sz_s_v, block_sz_s_v, sizeof(float) * sm_sz_s_v>>>(d_cost, d_acost, d_cross, num_disp, num_cols, num_rows, sm_cols_s_v, sm_sz_s_v, sm_padding_s_v, ipt_s_v); 
    stopCudaTimer(&timer, "Cross Horizontal Transposed Sum Kernel #2");
	
	startCudaTimer(&timer);
    ca_cross_vhsum_kernel_2<<<grid_sz_s_v, block_sz_s_v, sizeof(float) * sm_sz_s_v>>>(d_acost, d_cost, d_cross, num_disp, num_cols, num_rows, sm_cols_s_v, sm_sz_s_v, sm_padding_s_v, ipt_s_v); 
    stopCudaTimer(&timer, "Cross Horizontal Transposed Sum Kernel");
	
    startCudaTimer(&timer);
    cost_transpose_kernel_4<<<grid_sz_t_v, block_sz_t_v>>>(d_cost, d_acost, num_disp, num_cols, num_rows, sm_width, sm_width); stopCudaTimer(&timer, "Cost Transpose Kernel #4");
	
	startCudaTimer(&timer);
    ca_cross_hsum_kernel_3<<<grid_sz_s, block_sz_s, sizeof(float) * sm_sz_s>>>(d_acost, d_cost, d_cross, num_disp, num_rows, num_cols, sm_cols_s, sm_sz_s, sm_padding_s, ipt_s); 
    stopCudaTimer(&timer, "Cross Horizontal Sum #3");
	
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMemcpy(acost[d], h_cost[d], sizeof(float) * num_cols * num_rows, cudaMemcpyDeviceToHost));
    }
    
     ///////////////////
    // DE-ALLOCATION // 
    ///////////////////

    cudaFree(d_img);
    cudaFree(d_cross);
    cudaFree(d_cost);
    cudaFree(d_acost);
    for (int d = 0; d < num_disp; ++d)
    {
        cudaFree(h_cost[d]);
        cudaFree(h_acost[d]);
    }
    for (int i = 0; i < CROSS_ARM_COUNT; ++i)
    {
        cudaFree(h_cross[i]);
    }
    free(h_cost);
    free(h_acost);
    free(h_cross);
}

#endif
