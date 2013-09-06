#ifndef D_IO_KERNEL 
#define D_IO_KERNEL
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "d_io.h"
#include "cuda_utils.h"
#include <math.h>

void adcensus_stm(unsigned char *img_sbs, float *disp_l, float *disp_r,
                  unsigned char** views, unsigned char* interlaced,
                  int num_rows, int num_cols_sbs, int num_cols, 
                  int num_rows_out, int num_cols_out, int elem_sz,
                  int num_views, int angle,
                  int num_disp, int zero_disp,
                  float ad_coeff, float census_coeff,
                  float ucd, float lcd, int usd, int lsd)
{
    ///////////////////////
    // MEMORY ALLOCATION //
    ///////////////////////

    size_t bw = 32;
    size_t bh = 32;
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
    
    size_t gw_sbs = (num_cols_sbs + bw - 1) / bw;
    const dim3 grid_sz_sbs(gw_sbs, gh, 1);
    
    int smem_w = bw + num_disp;
    int smem_sz = smem_w * bh * elem_sz; 

    ///////////////////////
    // MEMORY ALLOCATION //
    ///////////////////////

    unsigned char* d_img_sbs;    
    unsigned char* d_img_l;
    unsigned char* d_img_r;

    checkCudaError(cudaMalloc(&d_img_sbs, sizeof(unsigned char) * num_rows * num_cols_sbs * elem_sz));
    checkCudaError(cudaMemcpy(d_img_sbs, img_sbs, sizeof(unsigned char) * num_rows * num_cols_sbs * elem_sz, cudaMemcpyHostToDevice)); 

    checkCudaError(cudaMalloc(&d_img_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMalloc(&d_img_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    
    ///////////////////////////
    // SIDE BY SIDE SPLITTER //
    ///////////////////////////

    demux_sbs<<<grid_sz_sbs, block_sz>>>(d_img_l, d_img_r, d_img_sbs, num_rows, num_cols_sbs, num_cols, elem_sz);
    cudaDeviceSynchronize();
    
    checkCudaError(cudaMemcpy(views[0], d_img_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(views[num_views - 1], d_img_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyDeviceToHost));

    cudaFree(d_img_sbs);

    /////////////////////////
    // COST INITIALIZATION //
    /////////////////////////
    
    float** d_adcensus_cost_l;
    float** d_adcensus_cost_r;
    
    checkCudaError(cudaMalloc(&d_adcensus_cost_l, sizeof(float*) * num_disp));
    checkCudaError(cudaMalloc(&d_adcensus_cost_r, sizeof(float*) * num_disp));
   
    float** h_adcensus_cost_l = (float**) malloc(sizeof(float*) * num_disp);
    float** h_adcensus_cost_r = (float**) malloc(sizeof(float*) * num_disp);

    d_ci_adcensus(d_img_l, d_img_r, d_adcensus_cost_l, d_adcensus_cost_r, 
                  h_adcensus_cost_l, h_adcensus_cost_r, ad_coeff, census_coeff, 
                  num_disp, zero_disp, num_rows, num_cols, elem_sz);
    
    //////////////////////
    // COST AGGRAGATION //
    //////////////////////

    float** d_acost_l, **d_acost_r;

    checkCudaError(cudaMalloc(&d_acost_l, sizeof(float*) * num_disp));
    checkCudaError(cudaMalloc(&d_acost_r, sizeof(float*) * num_disp));

    float** h_acost_l = (float**) malloc(sizeof(float*) * num_disp);
    float** h_acost_r = (float**) malloc(sizeof(float*) * num_disp);
    
    d_ca_cross(d_img_l, d_adcensus_cost_l, h_adcensus_cost_l, d_acost_l, h_acost_l, ucd, lcd, usd, lsd, num_disp, num_rows, num_cols, elem_sz);
    
    d_ca_cross(d_img_r, d_adcensus_cost_r, h_adcensus_cost_r, d_acost_r, h_acost_r, ucd, lcd, usd, lsd, num_disp, num_rows, num_cols, elem_sz);

    cudaFree(d_adcensus_cost_l);
    cudaFree(d_adcensus_cost_r);
    for (int d = 0; d < num_disp; ++d)
    {
        cudaFree(h_adcensus_cost_l[d]);
        cudaFree(h_adcensus_cost_r[d]);
    }
    free(h_adcensus_cost_l); 
    free(h_adcensus_cost_r); 
    
    ///////////////////////////
    // DISPARITY COMPUTATION //
    ///////////////////////////
    float* d_disp_raw_l, *d_disp_raw_r;
    
    checkCudaError(cudaMalloc(&d_disp_raw_l, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMalloc(&d_disp_raw_r, sizeof(float) * num_rows * num_cols));
    
    d_dc_wta(d_acost_l, d_disp_raw_l, num_disp, zero_disp, num_rows, num_cols);
    d_dc_wta(d_acost_r, d_disp_raw_r, num_disp, zero_disp, num_rows, num_cols);

    float* d_disp_l, *d_disp_r;
    
    checkCudaError(cudaMalloc(&d_disp_l, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMalloc(&d_disp_r, sizeof(float) * num_rows * num_cols));
    
    d_filter_bilateral_1(d_disp_l, d_disp_raw_l, 7, 5, 10, num_rows, num_cols);
    d_filter_bilateral_1(d_disp_r, d_disp_raw_r, 7, 5, 10, num_rows, num_cols);

    checkCudaError(cudaMemcpy(disp_l, d_disp_l, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(disp_r, d_disp_r, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
     
    cudaFree(d_acost_l);
    cudaFree(d_acost_r);
    cudaFree(d_disp_raw_l);
    cudaFree(d_disp_raw_r);
    for (int d = 0; d < num_disp; ++d)
    {
        cudaFree(h_acost_l[d]);
        cudaFree(h_acost_r[d]);
    }
    free(h_acost_l); 
    free(h_acost_r); 
    
    //////////
    // DIBR //
    //////////

    unsigned char *d_occl_raw_l, *d_occl_raw_r;
    
    checkCudaError(cudaMalloc(&d_occl_raw_l, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMalloc(&d_occl_raw_r, sizeof(unsigned char) * num_rows * num_cols));
    
    d_dibr_occl(d_occl_raw_l, d_occl_raw_r, d_disp_l, d_disp_r, num_rows, num_cols);
    
    unsigned char *d_occl_l, *d_occl_r;
    
    checkCudaError(cudaMalloc(&d_occl_l, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMalloc(&d_occl_r, sizeof(unsigned char) * num_rows * num_cols));

    d_filter_bleed_1(d_occl_l, d_occl_raw_l, 1, num_rows, num_cols);    
    d_filter_bleed_1(d_occl_r, d_occl_raw_r, 1, num_rows, num_cols);    
     
    unsigned char** h_views = (unsigned char**) malloc(sizeof(unsigned char*) * num_views);
    h_views[0] = d_img_r;
    h_views[num_views - 1] = d_img_l;
    for (int v = 1; v < num_views - 1; ++v)
        checkCudaError(cudaMalloc(&h_views[v], sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    
    for (int v = 1; v < num_views - 1; ++v)
    {   
        float shift = 1.0 - ((1.0 * (float) v) / ((float) num_views - 1.0));
        d_dibr_dbm(h_views[v], d_img_l, d_img_r, d_disp_l, d_disp_r, d_occl_l, d_occl_r, shift, num_rows, num_cols, elem_sz);
    }

    for (int v = 1; v <  num_views - 1; ++v)
        checkCudaError(cudaMemcpy(views[v], h_views[v], sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyDeviceToHost));

    /////////
    // MUX //
    /////////
    unsigned char** d_views;
    checkCudaError(cudaMalloc(&d_views, sizeof(unsigned char*) * num_views));
    checkCudaError(cudaMemcpy(d_views, h_views, sizeof(unsigned char*) * num_views, cudaMemcpyHostToDevice));

    unsigned char* d_interlaced;
    checkCudaError(cudaMalloc(&d_interlaced, sizeof(unsigned char) * num_rows_out * num_cols_out * elem_sz));

    d_mux_multiview(d_views, d_interlaced, num_views, angle, num_rows, num_cols, num_rows_out, num_cols_out, elem_sz);

    checkCudaError(cudaMemcpy(interlaced, d_interlaced, sizeof(unsigned char) * num_rows_out * num_cols_out * elem_sz, cudaMemcpyDeviceToHost));
    
    ///////////////////
    // DE-ALLOCATION //
    ///////////////////
    
    cudaFree(d_img_l);
    cudaFree(d_img_r);
    cudaFree(d_disp_l);
    cudaFree(d_disp_r);
    cudaFree(d_occl_l);
    cudaFree(d_occl_r);
    for (int v = 1; v < num_views - 1; ++v)
        cudaFree(h_views[v]);
    free(h_views);
    cudaFree(d_views);
    cudaFree(d_interlaced);
}


#endif
