#ifndef D_TX_SCALE_TEX_H
#define D_TX_SCALE_TEX_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void d_tx_scale_tex(unsigned char* in_data, unsigned char* out_data, int elem_sz, int width_step, 
                    int in_rows, int in_cols, int out_rows, int out_cols);

#endif
