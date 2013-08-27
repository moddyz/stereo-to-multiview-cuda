#ifndef D_MUX_MULTIVIEW_KERNEL
#define D_MUX_MULTIVIEW_KERNEL
#include "d_mux_multiview.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void mux_multiview_kernel(unsigned char** views, unsigned char* output, float angle, int num_views,
									int in_rows, int in_cols, int out_rows, int out_cols, int elem_sz)
{
    // Thread Id's
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx >= out_cols || ty >= out_rows)
        return;
    
    float PI = 3.141592653589793;
    
    // Compute Input Sampling Coordinates
    float x_samp = fmin(fmax(((float) tx / (float) out_cols) * (float) in_cols, 0), (float) (in_cols - 1));
    float y_samp = fmin(fmax(((float) ty / (float) out_rows) * (float) in_rows, 0), (float) (in_rows - 1));
    
    // Interlace Specific
    float x_interval = num_views;
    float y_interval = (float) num_views / tan(angle * PI / 180.0) / 3.0;
    float y_view = ty % ((int) round(y_interval)) + 1.0;
    y_view = y_view * x_interval / y_interval;
    int x_view = (tx * 3 + (int) y_view) % ((int) x_interval);
    int r_view = x_view;
    if (r_view < 0)
        r_view = r_view + num_views;
    int g_view = r_view + 1;
    int b_view = r_view + 2;
    if (g_view >= num_views)
        g_view = g_view - num_views;
    if (b_view >= num_views)
        b_view =  b_view - num_views;
    if (r_view < 0 || g_view < 0 ||  b_view < 0)
        printf("r:%d g:%d b:%d\n", r_view, g_view, b_view);

    // Write to Output
    int b_out = (tx + ty * out_cols) * elem_sz;
    int g_out = b_out + 1;
    int r_out = g_out + 1;

    output[b_out] = alu_bilinear_interp(views[b_view], elem_sz, 0, x_samp, y_samp, in_cols, in_rows);
    output[g_out] = alu_bilinear_interp(views[g_view], elem_sz, 1, x_samp, y_samp, in_cols, in_rows);
    output[r_out] = alu_bilinear_interp(views[r_view], elem_sz, 2, x_samp, y_samp, in_cols, in_rows);
}

void d_mux_multiview( unsigned char **views, unsigned char* out_data, int num_views, float angle, 
				      int in_width, int in_height, int out_width, int out_height, int elem_sz)
{
    cudaEventPair_t timer;
	// Memory Allocation of Input
	unsigned char** d_views;
	checkCudaError(cudaMalloc(&d_views, sizeof(unsigned char *) * num_views));
	
	unsigned char** h_views = (unsigned char**) malloc(sizeof(unsigned char**) * num_views);
	for (int v = 0; v < num_views; ++v)
	{
		checkCudaError(cudaMalloc(&h_views[v], sizeof(unsigned char) * in_width * in_height * elem_sz));
		checkCudaError(cudaMemcpy(h_views[v], views[v], sizeof(unsigned char) * in_width * in_height * elem_sz, cudaMemcpyHostToDevice));
	}
	checkCudaError(cudaMemcpy(d_views, h_views, sizeof(unsigned char *) * num_views, cudaMemcpyHostToDevice));

	// Memory Allocation of Output
	unsigned char* d_out_data;
	checkCudaError(cudaMalloc(&d_out_data, sizeof(unsigned char) * out_width * out_height * elem_sz));
    
	// Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    
    size_t gw = (out_width + bw - 1) / bw;
    size_t gh = (out_height + bh - 1) / bh;
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
 	
    // Launch Kernel
    startCudaTimer(&timer);
 	mux_multiview_kernel<<<grid_sz, block_sz>>>(d_views, d_out_data, angle, num_views, in_height, in_width, out_height, out_width, elem_sz);
    cudaDeviceSynchronize();
    stopCudaTimer(&timer, "Multiview Interlace Kernel");

	// Copy Memory back to Host
	checkCudaError(cudaMemcpy(out_data, d_out_data, sizeof(unsigned char) * out_width * out_height * elem_sz,cudaMemcpyDeviceToHost));

	// De-allocation of Host & Device Memory
	for (int v = 0; v < num_views; ++v)
		cudaFree(h_views[v]);
	cudaFree(d_views);
	cudaFree(d_out_data);
    free(h_views);
}

#endif
