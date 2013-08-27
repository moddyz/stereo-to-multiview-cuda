#ifndef D_TX_SCALE_TEXTURE_KERNEL
#define D_TX_SCALE_TEXTURE_KERNEL

#include "d_tx_scale.h"
#include "cuda_utils.h"
#include <math.h>

texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void tx_scale_texture_kernel(unsigned char* out_data, int elem_sz, 
                                        int in_rows, int in_cols, int out_rows, int out_cols)
{
    // Thread Id's
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx >= out_cols || ty >= out_rows)
        return;
    
    // Compute Input Sampling Coordinates
    float x_samp = fmin(fmax(((float) tx / (float) out_cols) * (float) in_cols, 0), (float) (in_cols - 1));
    float y_samp = fmin(fmax(((float) ty / (float) out_rows) * (float) in_rows, 0), (float) (in_rows - 1));
    
    // Write to Output
    int b_out = (tx + ty * out_cols) * elem_sz;
    int g_out = b_out + 1;
    int r_out = g_out + 1;

    out_data[b_out] = tex2D(texRef, (3 * x_samp), y_samp);
    out_data[g_out] = tex2D(texRef, (3 * x_samp) + 1, y_samp);
    out_data[r_out] = tex2D(texRef, (3 * x_samp) + 2, y_samp);
}

void d_tx_scale_tex(unsigned char* in_data, unsigned char* out_data, int elem_sz, int width_step, 
                    int in_rows, int in_cols, int out_rows, int out_cols)
{
    // Device Output Memory Allocation
    unsigned char* d_out_data;
    checkCudaError(cudaMalloc(&d_out_data, sizeof(unsigned char) * out_rows * out_cols * elem_sz));
    
    // Device Texture Allocation & Data Copy
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaArray* d_in_data;
    checkCudaError(cudaMallocArray(&d_in_data, &channelDesc, in_cols * elem_sz, in_rows));
    checkCudaError(cudaMemcpy2DToArray(d_in_data, 0, 0, in_data, width_step, 
                                       in_cols * elem_sz * sizeof(unsigned int), in_rows, cudaMemcpyHostToDevice));

    cudaBindTextureToArray(texRef, d_in_data, channelDesc); 
 
    // Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    
    size_t gw = (out_cols + bw - 1) / bw;
    size_t gh = (out_rows + bh - 1) / bh;
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
   
    // Launch Kernel
    startCudaTimer(&timer);
    tx_scale_texture_kernel<<<grid_sz, block_sz>>>(d_out_data, elem_sz, 
                                                               in_rows, in_cols, out_rows, out_cols);
    stopCudaTimer(&timer, "Scale Kernel - Texture"); 
    
    // Copy Data Device -> Host
    checkCudaError(cudaMemcpy(out_data, d_out_data, 
                   sizeof(unsigned char) * out_rows * out_cols * elem_sz, cudaMemcpyDeviceToHost));

    // Device Memory De-allocation
    cudaFree(d_in_data);
    cudaFree(d_out_data);

}

#endif
