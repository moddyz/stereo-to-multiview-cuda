#ifndef D_MUX_MULTIVIEW_H 
#define D_MUX_MULTIVIEW_H 
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "d_alu.h"

__global__ void mux_multiview_kernel(unsigned char** views, unsigned char* output, 
                                    int num_views, float angle, 
									int in_rows, int in_cols, int out_rows, int out_cols, int elem_sz);

void d_mux_multiview( unsigned char **views, unsigned char* out_data, 
                      int num_views, float angle, 
				      int in_rows, int in_cols, int out_rows, int out_cols, int elem_sz);

#endif
