#ifndef D_DC_HSLO_KERNEL 
#define D_DC_HSLO_KERNEL
#include "d_dc_wta.h"
#include "cuda_utils.h"
#include <math.h>
#include <float.h>
#include <limits.h>

__global__ void dc_hslo_v_cost_kernel(float** cost, float** pcost, float* disp, 
                                      int num_disp, int num_rows, int num_cols)
{

}

__global__ void dc_hslo_h_cost_kernel(float** cost, float** pcost, float* disp, 
                                     int num_disp, int num_rows, int num_cols)
{
    int r = blockIdx.x;
    int d = threadIdx.x;
    
    if (r > num_rows - 1)
        return;
    
    for (int p = 0; p < num_cols; ++p)
    {
        int pr = max(p - 1, 0); 
        float C1 = cost[d][p + r * num_cols];
    }
}

__global__ void dc_hslo_h_cdiff_kernel(float* P1_l, float** P2_l, 
                                       float* P1_r, float** P2_r, 
                                       unsigned char* img_l, unsigned char* img_r,
                                       int r_dir,
                                       float T, float H1, float H2, 
                                       float H1_4, float H2_4, 
                                       float H1_10, float H2_10,
                                       int num_disp, int zero_disp, 
                                       int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tx > (num_cols * num_rows) - 1)
        return;
    
    int r = r_dir;

    if (r == 0 && r_dir == -1)
        r = 0;
    
    if (tx + r == (num_cols * num_rows) - 1 && r_dir == 1)
        r = 0;

    for (int d = 0; d < num_disp; ++d)
    {
        int p_b_l = tx * elem_sz; 
        int pr_b_l = (tx - r) * elem_sz; 

        unsigned char p_cavg_l = (img_l[p_b_l] + img_l[p_b_l + 1] + img_l[p_b_l + 2]) / 3;
        unsigned char pr_cavg_l = (img_l[pr_b_l] + img_l[pr_b_l + 1] + img_l[pr_b_l + 2]) / 3;
       
        int dx = min(max(tx + d - zero_disp, 0), (num_cols * num_rows) - 1);
        int dxr = min(max(tx + r + d - zero_disp, 0), (num_cols * num_rows) - 1);
        int p_b_r = dx + elem_sz; 
        int pr_b_r = dxr + elem_sz; 
        
        float p_cavg_r = (float) (img_r[p_b_r] + img_r[p_b_r + 1] + img_r[p_b_r + 2]) / 3.0;
        float pr_cavg_r = (float) (img_r[pr_b_r] + img_r[pr_b_r + 1] + img_r[pr_b_r + 2]) / 3.0;
        
        float p_cdiff_l = abs(p_cavg_l - pr_cavg_l);
        float p_cdiff_r = abs(p_cavg_r - pr_cavg_r);

        if (p_cdiff_l < T && p_cdiff_r < T)
        {
            P1_l[tx] = H1;
            P2_l[d][tx] = H2;
            P1_r[dx] = H1;
            P2_r[d][dx] = H2;
        }
        else if ((p_cdiff_l < T && p_cdiff_r > T) || (p_cdiff_l > T && p_cdiff_r < T))
        {
            P1_l[tx] = H1_4;
            P2_l[d][tx] = H2_4;
            P1_r[dx] = H1_4;
            P2_r[d][dx] = H2_4;
        }
        else 
        {
            P1_l[tx] = H1_10;
            P2_l[d][tx] = H2_10;
            P1_r[dx] = H1_10;
            P2_r[d][dx] = H2_10;
        }
    }
}

void dc_hslo(float** cost, float* disp, 
             unsigned char* img_l, unsigned char* img_r,
             float T, float H1, float H2, 
             int num_disp, int zero_disp, 
             int num_rows, int num_cols, int elem_sz)
{
    cudaEventPair_t timer;
    
    /////////////////////// 
    // DEVICE PARAMETERS //
    ///////////////////////

    size_t tsz_h = num_disp;
    size_t gsz_h = num_rows;
    const dim3 block_sz_h(tsz_h, 1, 1);
    const dim3 grid_sz_h(gsz_h, 1, 1);
    
    size_t tsz_v = num_disp;
    size_t gsz_v = num_cols;
    const dim3 block_sz_v(tsz_v, 1, 1);
    const dim3 grid_sz_v(gsz_v, 1, 1);
    
    size_t bw = 1024;
    size_t gw = (num_cols * num_rows + bw - 1) / bw;
    const dim3 block_sz(bw, 1, 1);
    const dim3 grid_sz(gw, 1, 1);
    
    float H1_4 = H1/4.0;
    float H2_4 = H2/4.0;
    float H1_10 = H1/10.0;
    float H2_10 = H2/10.0;
    
    ///////////////////////
    // MEMORY ALLOCATION //
    ///////////////////////
    
    unsigned char* d_img_l, *d_img_r;

    checkCudaError(cudaMalloc(&d_img_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMalloc(&d_img_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    
    checkCudaError(cudaMemcpy(d_img_l, img_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_img_r, img_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));

    float* d_P1_l, *d_P1_r;

    checkCudaError(cudaMalloc(&d_P1_l, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMalloc(&d_P1_r, sizeof(float) * num_rows * num_cols));

    float** d_P2_l, **d_P2_r;
    
    checkCudaError(cudaMalloc(&d_P2_l, sizeof(float*) * num_disp));
    float** h_P2_l = (float**) malloc(sizeof(float*) * num_disp);
    for (int d = 0; d < num_disp; ++d)
        checkCudaError(cudaMalloc(&h_P2_l[d], sizeof(float) * num_rows * num_cols));

    checkCudaError(cudaMemcpy(d_P2_l, h_P2_l, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    
    checkCudaError(cudaMalloc(&d_P2_r, sizeof(float*) * num_disp));
    float** h_P2_r = (float**) malloc(sizeof(float*) * num_disp);
    for (int d = 0; d < num_disp; ++d)
        checkCudaError(cudaMalloc(&h_P2_r[d], sizeof(float) * num_rows * num_cols));

    checkCudaError(cudaMemcpy(d_P2_r, h_P2_r, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    
    startCudaTimer(&timer); 
    dc_hslo_h_cdiff_kernel<<<grid_sz, block_sz>>>(d_P1_l, d_P2_l, d_P1_r, d_P2_r, d_img_l, d_img_r, -1, T, H1, H2, H1_4, H2_4, H1_10, H2_10, num_disp, zero_disp, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Scanline Optimization H-R CDIFF Kernel"); 
    
    float** d_cost;

    checkCudaError(cudaMalloc(&d_cost, sizeof(float*) * num_disp));

    float** h_cost = (float**) malloc(sizeof(float*) * num_disp);
    
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMalloc(&h_cost[d], sizeof(float) * num_rows * num_cols));
        checkCudaError(cudaMemcpy(h_cost[d], cost[d], sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));
    }

    checkCudaError(cudaMemcpy(d_cost, h_cost, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));


    float** d_pcost;

    checkCudaError(cudaMalloc(&d_pcost, sizeof(float*) * num_disp));

    float** h_pcost = (float**) malloc(sizeof(float*) * num_disp);
    
    for (int d = 0; d < num_disp; ++d)
        checkCudaError(cudaMalloc(&h_pcost[d], sizeof(float) * num_rows * num_cols));

    checkCudaError(cudaMemcpy(d_pcost, h_pcost, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));



    float* d_disp;
    checkCudaError(cudaMalloc(&d_disp, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMemset(d_disp, 0, sizeof(float) * num_rows * num_cols));
    
    
    ///////////////////
    // DE-ALLOCATION //
    ///////////////////
    
    cudaFree(d_img_l);
    cudaFree(d_img_r);
    cudaFree(d_P1_l);
    cudaFree(d_P2_l);
    cudaFree(d_P1_r);
    cudaFree(d_P2_r);
    cudaFree(d_cost);
    cudaFree(d_pcost);
    cudaFree(d_disp);
    for (int d = 0; d < num_disp; ++d)
    {
        cudaFree(h_cost[d]);
        cudaFree(h_pcost[d]);
        cudaFree(h_P2_l[d]);
        cudaFree(h_P2_r[d]);
    }
    free(h_cost);
    free(h_pcost);
}

#endif
