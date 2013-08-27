#ifndef D_MUX_MULTIVIEW_H 
#define D_MUX_MULTIVIEW_H 
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "d_alu.h"

void d_mux_multiview( unsigned char **views, unsigned char* out_data, int num_views, float angle, 
				      int in_width, int in_height, int out_width, int out_height, int elem_sz);

#endif
