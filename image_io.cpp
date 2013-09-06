#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cvaux.h>
#include "d_dibr_occl.h"
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
    DISPLAY_MODE_OCCLUSION,
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

int main(int argc, char** argv)
{
    if (argc != 15) 
    {
        printf("Place images in img subdir: \n");
        printf("then input file names directly w/o dir extension \n");
        printf("Usage: ./program [left file] [right file] [ad coeff] [census coeff] [ndisp] [zerodisp] [upper color limit] [lower color limit] [upper spatial limit] [lower spatial limit] [num views] [angle] [out width] [out height]\n");
        return -1;
    } 
    
    printDeviceInfo();
    
    printf("=======================================\n");
    printf("== STEREO TO MULTIVIEW IMAGE PROCESS ==\n");
    printf("=======================================\n\n");
    
    ////////////////// 
    // FILE PARSING //
    //////////////////
    
	const char* imgdir = "./img/";
    char* file_l = argv[1];
    char *full_file_l = (char *) malloc(snprintf(NULL, 0, "%s%s.bmp", imgdir, file_l) + 1); 
    sprintf(full_file_l, "%s%s.bmp", imgdir, file_l);
    printf("Reading %s...\n", full_file_l);
    
    char* file_r = argv[2];
    char *full_file_r = (char *) malloc(snprintf(NULL, 0, "%s%s.bmp", imgdir, file_r) + 1); 
    sprintf(full_file_r, "%s%s.bmp", imgdir, file_r);
    printf("Reading %s...\n", full_file_r);

    ///////////////// 
    // READ IMAGES //
    /////////////////

    Mat mat_img_l = imread(full_file_l, CV_LOAD_IMAGE_COLOR);
    Mat mat_img_r = imread(full_file_r, CV_LOAD_IMAGE_COLOR);

    free(full_file_l);
    free(full_file_r);
    
	if (mat_img_l.empty() || mat_img_r.empty())
    {
        printf("Error! Could not read image files from disk! \n");
        return -1;
    }

    unsigned char* data_img_l = mat_img_l.data;
    unsigned char* data_img_r = mat_img_r.data;
    
    int num_rows = mat_img_l.rows;
    int num_cols = mat_img_l.cols;
    int elem_sz  = mat_img_l.elemSize();
    
    ///////////////////// 
    // READ PARAMETERS //
    /////////////////////
    
	float ad_coeff = atof(argv[3]);
    float census_coeff = atof(argv[4]);
    int num_disp = atoi(argv[5]);
    int zero_disp = atoi(argv[6]);
	float ucd = atof(argv[7]);
	float lcd = atof(argv[8]);
	int usd = atoi(argv[9]);
	int lsd = atoi(argv[10]);
	int num_views = atoi(argv[11]);
	float angle = atof(argv[12]);
	int num_cols_out = atoi(argv[13]);
	int num_rows_out = atoi(argv[14]);

    printf("Input Width:             %d\n", num_cols); 
    printf("Input Height:            %d\n", num_rows); 
    printf("Number of Views:         %d\n", num_views);
    printf("Angle of Attenuator:     %f\n", angle);
    printf("Output Width:            %d\n", num_cols_out); 
    printf("Output Height:           %d\n", num_rows_out); 
    printf("Number of Disparities:   %d\n", num_disp);
    printf("Zero Disparity Index:    %d\n", zero_disp);
    printf("AD Coefficient:          %f\n", ad_coeff);
    printf("Census Coefficient:      %f\n", census_coeff);
    printf("Upper Color Delta:       %f\n", ucd);
    printf("Lower Color Delta:       %f\n", lcd);
    printf("Upper Spatial Delta:     %d\n", usd);
    printf("Lower Spatial Delta:     %d\n", lsd);
    printf("\n");
    
    /////////////////////////
    // COST INITIALIZATION //
    /////////////////////////
   
	std::vector<Mat> mat_cost_l;
    float ** data_cost_l = (float**) malloc(sizeof(float*) * num_disp);
    
	for (int d = 0; d < num_disp; ++d)
        mat_cost_l.push_back(Mat::zeros(num_rows, num_cols, CV_32F));
    for (int d = 0; d < num_disp; ++d)
        data_cost_l[d] = (float*) mat_cost_l[d].data;

    std::vector<Mat> mat_cost_r;
    float** data_cost_r = (float**) malloc(sizeof(float*) * num_disp);
    
	for (int d = 0; d < num_disp; ++d)
        mat_cost_r.push_back(Mat::ones(num_rows, num_cols, CV_32F));
    for (int d = 0; d < num_disp; ++d)
        data_cost_r[d] = (float*) mat_cost_r[d].data;
    
    ci_adcensus(data_img_l, data_img_r, data_cost_l, data_cost_r, ad_coeff, census_coeff, num_disp, zero_disp, num_rows, num_cols, elem_sz);
	
    //////////////////////
    // COST AGGRAGATION //
    //////////////////////
    
	std::vector<Mat> mat_acost_l;
    float ** data_acost_l = (float**) malloc(sizeof(float*) * num_disp);
    
	for (int d = 0; d < num_disp; ++d)
        mat_acost_l.push_back(Mat::zeros(num_rows, num_cols, CV_32F));
    for (int d = 0; d < num_disp; ++d)
        data_acost_l[d] = (float*) mat_acost_l[d].data;

    std::vector<Mat> mat_acost_r;
    float** data_acost_r = (float**) malloc(sizeof(float*) * num_disp);
    
	for (int d = 0; d < num_disp; ++d)
    {
        mat_acost_r.push_back(Mat::zeros(num_rows, num_cols, CV_32F));
    }
    for (int d = 0; d < num_disp; ++d)
        data_acost_r[d] = (float*) mat_acost_r[d].data;
	
	ca_cross(data_img_l, data_cost_l, data_acost_l, ucd, lcd, usd, lsd, num_disp, num_rows, num_cols, elem_sz);

	ca_cross(data_img_r, data_cost_r, data_acost_r, ucd, lcd, usd, lsd, num_disp, num_rows, num_cols, elem_sz);

    ///////////////////////////
    // DISPARITY COMPUTATION //
    ///////////////////////////
	
	Mat mat_disp_l = Mat::zeros(num_rows, num_cols, CV_32F);
	Mat mat_disp_r = Mat::zeros(num_rows, num_cols, CV_32F);

	float* data_disp_l = (float*) mat_disp_l.data;
	float* data_disp_r = (float*) mat_disp_r.data;

	dc_wta(data_acost_l, data_disp_l, num_disp, zero_disp, num_rows, num_cols);
	dc_wta(data_acost_r, data_disp_r, num_disp, zero_disp, num_rows, num_cols);
    
	//////////
    // DIBR //
    //////////

    Mat mat_occl_l = Mat::zeros(num_rows, num_cols, CV_8UC(1));
    Mat mat_occl_r = Mat::zeros(num_rows, num_cols, CV_8UC(1));
    
    unsigned char* data_occl_l = mat_occl_l.data;
    unsigned char* data_occl_r = mat_occl_r.data;

    dibr_occl(data_occl_l, data_occl_r, data_disp_l, data_disp_r, num_rows, num_cols);
	
    std::vector<Mat> mat_views;
	mat_views.push_back(mat_img_r);
    for (int v = 1; v < num_views - 1; ++v)
		mat_views.push_back(Mat::zeros(num_rows, num_cols, CV_8UC(3)));
	mat_views.push_back(mat_img_l);

    unsigned char **data_views = (unsigned char **) malloc(sizeof(unsigned char *) * num_views);
    
	for (int v = 0; v < num_views; ++v)
        data_views[v] = mat_views[v].data;
    
    for (int v = 1; v < num_views - 1; ++v)
    {
        float shift = 1.0 - ((1.0 * (float) v) / ((float) num_views - 1.0));
        dibr_dfm(data_views[v], data_img_l, data_img_r, data_disp_l, data_disp_r, shift, num_rows, num_cols, elem_sz);
    }

	/////////
    // MUX //
    /////////

    Mat mat_mux = Mat::zeros(num_rows_out, num_cols_out, CV_8UC(3));
    unsigned char* data_mux = mat_mux.data;
    
    mux_multiview(data_views, data_mux, num_views, angle, num_rows, num_cols, num_rows_out, num_cols_out, elem_sz);

	// Equalize Images for Display
    for (int d = 0; d < num_disp; ++d)
    {
        normalize(mat_cost_l[d], mat_cost_l[d], 0, 1, CV_MINMAX);
        normalize(mat_cost_r[d], mat_cost_r[d], 0, 1, CV_MINMAX);
        normalize(mat_acost_l[d], mat_acost_l[d], 0, 1, CV_MINMAX);
        normalize(mat_acost_r[d], mat_acost_r[d], 0, 1, CV_MINMAX);
    }
	normalize(mat_disp_l, mat_disp_l, 0, 1, CV_MINMAX);
	normalize(mat_disp_r, mat_disp_r, 0, 1, CV_MINMAX);
    normalize(mat_occl_l, mat_occl_l, 0, 255, CV_MINMAX);
    normalize(mat_occl_r, mat_occl_r, 0, 255, CV_MINMAX);
   
    //////////////////
    // TESTING HSLO //
    //////////////////
/*
    float T = 15.0;
    float H1 = 1.0;
    float H2 = 3.0;

    dc_hslo(data_cost_l, data_disp_l, data_img_l, data_img_r, T, H1, H2, num_disp, zero_disp, num_rows, num_cols, elem_sz); 
 */   
    /////////////
    // DISPLAY //
    /////////////
    
	int display_mode = DISPLAY_MODE_COST;
    int display_persp = 0;
    int disp_level = zero_disp - 1;
	int display_view = 0;
    
	namedWindow("Display");
    imshow("Display", mat_cost_l[disp_level]);
	
	while( 1 )
    {
        char key = waitKey(0);

		// Handle Keys
		switch (key)
		{
			case 'q':
            	display_persp = DISPLAY_PERSP_LEFT;
				break;
			case 'w':
            	display_persp = DISPLAY_PERSP_RIGHT;
				break;
			case '1':
            	display_mode = DISPLAY_MODE_SOURCE;
				break;
			case '2':
            	display_mode = DISPLAY_MODE_COST;
				break;
			case '3':
            	display_mode = DISPLAY_MODE_ACOST;
				break;
			case '4':
            	display_mode = DISPLAY_MODE_DISPARITY;
				break;
			case '5':
            	display_mode = DISPLAY_MODE_OCCLUSION;
				break;
			case '6':
            	display_mode = DISPLAY_MODE_MULTIVIEW;
				break;
			case '7':
            	display_mode = DISPLAY_MODE_INTERLACED;
				break;
			case ']':
				if (display_mode == DISPLAY_MODE_COST ||
					display_mode == DISPLAY_MODE_ACOST)
					disp_level = min(disp_level + 1, num_disp - 1);
				else if (display_mode == DISPLAY_MODE_MULTIVIEW)
					display_view = min(display_view + 1, num_views - 1);
				break;
			case '[':
				if (display_mode == DISPLAY_MODE_COST ||
					display_mode == DISPLAY_MODE_ACOST)
					disp_level = max(disp_level - 1, 0);
				else if (display_mode == DISPLAY_MODE_MULTIVIEW)
					display_view = max(display_view - 1, 0);
				break;
			default:
				break;
		}
		
		// Handle Display
		switch (display_mode)
		{
			case DISPLAY_MODE_SOURCE:
				if (display_persp == DISPLAY_PERSP_LEFT)
				{
					imshow("Display", mat_img_l);
					printf("Displaying Left Source\n");
				}
				else if (display_persp == DISPLAY_PERSP_RIGHT)
				{
					imshow("Display", mat_img_r);
					printf("Displaying Right Source\n");
				}
				break;
			case DISPLAY_MODE_COST:
				if (display_persp == DISPLAY_PERSP_LEFT)
				{
					imshow("Display", mat_cost_l[disp_level]);
					printf("Showing Left Cost Disparity Level: %d\n", disp_level - zero_disp + 1);
				}
				else if (display_persp == DISPLAY_PERSP_RIGHT)
				{
					imshow("Display", mat_cost_r[disp_level]);
					printf("Showing Right Cost Disparity Level: %d\n", disp_level - zero_disp + 1);
				}
				break;
			case DISPLAY_MODE_ACOST:
				if (display_persp == DISPLAY_PERSP_LEFT)
				{
					imshow("Display", mat_acost_l[disp_level]);
					printf("Showing Left Aggragated Cost Disparity Level: %d\n", disp_level - zero_disp + 1);
				}
				else if (display_persp == DISPLAY_PERSP_RIGHT)
				{
					imshow("Display", mat_acost_r[disp_level]);
					printf("Showing Right Aggragated Cost Disparity Level: %d\n", disp_level - zero_disp + 1);
				}
				break;
			case DISPLAY_MODE_DISPARITY:
				if (display_persp == DISPLAY_PERSP_LEFT)
				{
					imshow("Display", mat_disp_l);
					printf("Showing Left Disparity\n");
				}
				else if (display_persp == DISPLAY_PERSP_RIGHT)
				{
					imshow("Display", mat_disp_r);
					printf("Showing Right Disparity\n");
				}
				break;
			case DISPLAY_MODE_OCCLUSION:
				if (display_persp == DISPLAY_PERSP_LEFT)
				{
					imshow("Display", mat_occl_l);
					printf("Showing Left Occlusion\n");
				}
				else if (display_persp == DISPLAY_PERSP_RIGHT)
				{
					imshow("Display", mat_occl_r);
					printf("Showing Right Occlusion\n");
				}
				break;
			case DISPLAY_MODE_MULTIVIEW:
				imshow("Display", mat_views[display_view]);
				printf("Showing View # %d\n", display_view + 1);
				break;
			case DISPLAY_MODE_INTERLACED:
				imshow("Display", mat_mux);
				printf("Showing Interlaced\n");
				break;
			default:
				break;
		}
    }

    free(data_cost_l); 
    free(data_cost_r); 
    free(data_acost_l); 
    free(data_acost_r); 
    mat_cost_l.clear();
    mat_cost_r.clear();
    mat_acost_l.clear();
    mat_acost_r.clear();
    mat_views.clear();
    return 0;
}
