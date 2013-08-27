#ifndef D_TX_SCALE_H
#define D_TX_SCALE_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "d_alu.h"

void d_tx_scale(unsigned char* in_data, unsigned char* out_data, int elem_sz, 
                int in_rows, int in_cols, int out_rows, int out_cols);

#endif

