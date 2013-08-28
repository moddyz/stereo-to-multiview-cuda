#ifndef D_CA_CROSS_KERNEL 
#define D_CA_CROSS_KERNEL
#include "d_ca_cross.h"
#include "cuda_utils.h"
#include <math.h>




void ca_cross(unsigned char* img_l, unsigned char* img_r, float** cost_l, float** cost_r,
              float** acost_l, float** acost_r, float ucd, float lcd, int usd, int lsd);
{
    cudaEventPair_t timer;
    
    //////////////////////// 
    // CROSS CONSTRUCTION //
    ////////////////////////

    unsigned char* d_img_l;
    unsigned char* d_img_r;

}

#endif
