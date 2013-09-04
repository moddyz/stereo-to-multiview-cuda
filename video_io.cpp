#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cvaux.h>
#include "d_dibr_warp.h"
#include "d_dc_wta.h"
#include "d_dc_hslo.h"
#include "d_ca_cross.h"
#include "d_ci_adcensus.h"
#include "d_ci_census.h"
#include "d_ci_ad.h"
#include "d_tx_scale.h"
#include "d_mux_multiview.h"
#include "cuda_utils.h"

using namespace cv;

typedef enum
{
	DISPLAY_PERSP_LEFT,
	DISPLAY_PERSP_RIGHT,
} display_persp_e;

typedef enum 
{
    DISPLAY_MODE_SOURCE,
    DISPLAY_MODE_COST,
    DISPLAY_MODE_ACOST,
    DISPLAY_MODE_DISPARITY,
    DISPLAY_MODE_MULTIVIEW,
    DISPLAY_MODE_INTERLACED,
} display_mode_e;

void printMatInfo(Mat mat, char *mat_name)
{
   int rows = mat.rows;
   int cols = mat.cols;
   int esz = mat.elemSize();

   printf("%s info:\n", mat_name);
   printf("Rows: %d, Cols: %d, Element Size: %d\n\n", rows, cols, esz);
}

int main( int argc, char **argv)
{

}
