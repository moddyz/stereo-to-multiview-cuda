#ifndef D_TX_SCALE_KERNEL
#define D_TX_SCALE_KERNEL
#include "d_tx_scale.h"
#include "d_alu.h"
#include "cuda_utils.h"
#include <math.h>


__global__ void tx_scale_bilinear_kernel(unsigned char* img_in, unsigned char* img_out,  
                                         int in_rows, int in_cols, int out_rows, int out_cols, int elem_sz)
{
    // Thread Id's
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;

    if (gx >= out_cols || gy >= out_rows)
        return;
    
    // Compute Input Sampling Coordinates
    float g_x_samp = fmin(fmax(((float) gx / (float) out_cols) * (float) in_cols, 0), (float) (in_cols - 1));
    float g_y_samp = fmin(fmax(((float) gy / (float) out_rows) * (float) in_rows, 0), (float) (in_rows - 1));
    
    // Write to Output
    int b_out = (gx + gy * out_cols) * elem_sz;
    int g_out = b_out + 1;
    int r_out = g_out + 1;

    img_out[b_out] = alu_bilinear_interp(img_in, elem_sz, 0, g_x_samp, g_y_samp, in_cols, in_rows);
    img_out[g_out] = alu_bilinear_interp(img_in, elem_sz, 1, g_x_samp, g_y_samp, in_cols, in_rows);
    img_out[r_out] = alu_bilinear_interp(img_in, elem_sz, 2, g_x_samp, g_y_samp, in_cols, in_rows);
}

__global__ void tx_scale_nearest_kernel(unsigned char* img_in, unsigned char* img_out,  
                                        int in_rows, int in_cols, int out_rows, int out_cols, int elem_sz)
{
    // Thread Id's
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx >= out_cols || ty >= out_rows)
        return;
    
    // Compute Input Sampling Coordinates
    float x_samp = fmin(fmax(((float) tx / (float) out_cols) * (float) in_cols, 0), (float) (in_cols - 1));
    float y_samp = fmin(fmax(((float) ty / (float) out_rows) * (float) in_rows, 0), (float) (in_rows - 1));
    
    int b_in = ((int) x_samp + (int) y_samp * in_cols) * elem_sz;
    int g_in = b_in + 1;
    int r_in = g_in + 1;

    // Write to Output
    int b_out = (tx + ty * out_cols) * elem_sz;
    int g_out = b_out + 1;
    int r_out = g_out + 1;

    img_out[b_out] = img_in[b_in];
    img_out[g_out] = img_in[g_in];
    img_out[r_out] = img_in[r_in];
}

void d_tx_scale(unsigned char* img_in, unsigned char* img_out, 
                int in_rows, int in_cols, int out_rows, int out_cols, int elem_sz)
{
    cudaEventPair_t timer;
    
    // Device Memory Allocation & Copy Data Host -> Device
    unsigned char* d_img_in;
    unsigned char* d_img_out;
    
    checkCudaError(cudaMalloc(&d_img_in, sizeof(unsigned char) * in_rows * in_cols * elem_sz));
    checkCudaError(cudaMemcpy(d_img_in, img_in, 
                   sizeof(unsigned char) * in_rows * in_cols * elem_sz, cudaMemcpyHostToDevice));
    
    checkCudaError(cudaMalloc(&d_img_out, sizeof(unsigned char) * out_rows * out_cols * elem_sz));

    // Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    
    size_t gw = (out_cols + bw - 1) / bw;
    size_t gh = (out_rows + bh - 1) / bh;
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);

    /* 
    float sw = ceil(((float) bw / (float) out_cols) * (float) in_cols);
    float sh = ceil(((float) bh / (float) out_rows) * (float) in_rows);
    const size_t shared_sz = sw * sh;
    printf("Shared Memory w:%f h:%f \n\n", sw, sh);
    */ 
    // Launch Kernel
    startCudaTimer(&timer);
    tx_scale_bilinear_kernel<<<grid_sz, block_sz>>>(d_img_in, d_img_out,
                                                    in_rows, in_cols, out_rows, out_cols, elem_sz);
                                                               
    stopCudaTimer(&timer, "Scale Kernel - Bilinear"); 
    
    // Copy Data Device -> Host
    checkCudaError(cudaMemcpy(img_out, d_img_out, 
                   sizeof(unsigned char) * out_rows * out_cols * elem_sz, cudaMemcpyDeviceToHost));

    // Device Memory De-allocation
    cudaFree(d_img_in);
    cudaFree(d_img_out);
}

#endif
