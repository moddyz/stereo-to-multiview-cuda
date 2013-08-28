#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cvaux.h>
#include "d_ci_adcensus.h"
#include "d_ci_census.h"
#include "d_ci_ad.h"
#include "d_tx_scale.h"
#include "d_mux_multiview.h"
#include "cuda_utils.h"

using namespace cv;

typedef enum 
{
    DISPLAY_SOURCE,
    DISPLAY_COST,
    DISPLAY_ACOST,
    DISPLAY_DISPARITY,
    DISPLAY_MULTIVIEW,
    DISPLAY_INTERLACED,
} display_type_e;

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
    if (argc != 7) 
    {
        printf("Place images in img subdir: \n");
        printf("then input file names directly w/o dir extension \n");
        printf("Usage: ./program [left file] [right file] [ad coeff] [census coeff] [ndisp] [zerodisp]\n");
        return -1;
    } 
    
    // File Path Parsing
    const char* imgdir = "./img/";
    char* file_l = argv[1];
    char *full_file_l = (char *) malloc(snprintf(NULL, 0, "%s%s.bmp", imgdir, file_l) + 1); 
    sprintf(full_file_l, "%s%s.bmp", imgdir, file_l);
    printf("Reading %s...\n", full_file_l);
    
    char* file_r = argv[2];
    char *full_file_r = (char *) malloc(snprintf(NULL, 0, "%s%s.bmp", imgdir, file_r) + 1); 
    sprintf(full_file_r, "%s%s.bmp", imgdir, file_r);
    printf("Reading %s...\n", full_file_r);

    // Parameter Parsing
    float ad_coeff = atof(argv[3]);
    float census_coeff = atof(argv[4]);
    int num_disp = atoi(argv[5]);
    int zero_disp = atoi(argv[6]);
    
    // Read Images
    Mat img_l = imread(full_file_l, CV_LOAD_IMAGE_COLOR);
    Mat img_r = imread(full_file_r, CV_LOAD_IMAGE_COLOR);
    free(full_file_l);
    free(full_file_r);
    if (img_l.empty() || img_r.empty())
    {
        printf("Error! Could not read image files from disk! \n");
        return -1;
    }

    unsigned char* data_img_l = img_l.data;
    unsigned char* data_img_r = img_r.data;
    
    int num_rows = img_l.rows;
    int num_cols = img_l.cols;
    int elem_sz  = img_l.elemSize();
    printf("rows: %d cols: %d\n", num_rows, num_cols);
    printf("adcoeff: %f censuscoeff: %f\n", ad_coeff, census_coeff);
    
    // Cost storage intialization
    std::vector<Mat> mat_cost_l;
    for (int d = 0; d < num_disp; ++d)
        mat_cost_l.push_back(Mat::zeros(num_rows, num_cols, CV_32F));
    float ** data_cost_l = (float**) malloc(sizeof(float*) * num_disp);
    for (int d = 0; d < num_disp; ++d)
        data_cost_l[d] = (float*) mat_cost_l[d].data;

    std::vector<Mat> mat_cost_r;
    for (int d = 0; d < num_disp; ++d)
    {
        mat_cost_r.push_back(Mat::ones(num_rows, num_cols, CV_32F));
    }
    float** data_cost_r = (float**) malloc(sizeof(float*) * num_disp);
    for (int d = 0; d < num_disp; ++d)
        data_cost_r[d] = (float*) mat_cost_r[d].data;
    
    // Cost Initiation
    printDeviceInfo();
    ci_adcensus(data_img_l, data_img_r, data_cost_l, data_cost_r, ad_coeff, census_coeff, num_disp, zero_disp, num_rows, num_cols, elem_sz);

    for (int d = 0; d < num_disp; ++d)
    {
        normalize(mat_cost_l[d], mat_cost_l[d], 0, 1, CV_MINMAX);
        normalize(mat_cost_r[d], mat_cost_r[d], 0, 1, CV_MINMAX);
    }

    // Display Images
    //int display_mode = DISPLAY_SOURCE;
    int disp_level = zero_disp - 1;
    int display_persp = 0;
    namedWindow("Display");
    imshow("Display", mat_cost_l[disp_level]);
	while( 1 )
    {
        char key = waitKey(0);
        if (key == '1')
            display_persp = 0;
        else if (key == '2')
            display_persp = 1;
        else if (key == '=')
            disp_level = min(disp_level + 1, num_disp - 1);
        else if (key == '-')
            disp_level = max(disp_level - 1, 0);
        if (display_persp == 0)
        {
            imshow("Display", mat_cost_l[disp_level]);
            printf("Showing Left Cost Disparity Level: %d\n", disp_level - zero_disp + 1);
        }
        else if (display_persp == 1)
        {
            imshow("Display", mat_cost_r[disp_level]);
            printf("Showing Right Cost Disparity Level: %d\n", disp_level - zero_disp + 1);
        }

    }

    free(data_cost_l); 
    free(data_cost_r); 
    mat_cost_l.clear();
    mat_cost_r.clear();
    return 0;
}

int multiview_routine( int argc, char **argv)
{
    // Parse Commands

    if (argc != 6) 
    {
        printf("Usage: ./program [file prefix] [num views] [angle] [output x] [output y]\n");
        return -1;
    }
    
    char * c_file_prefix = argv[1];
    float angle = atof(argv[3]);
    int num_views = atoi(argv[2]);
    int out_width = atoi(argv[4]);
    int out_height = atoi(argv[5]);

    std::vector<Mat> views;
    for (int v = 0; v < num_views; ++v)
    {
        char *file = (char *) malloc(snprintf(NULL, 0, "%s%d.bmp", c_file_prefix, v + 1) + 1); 
        sprintf(file, "%s%d.bmp", c_file_prefix, v + 1);
        printf("Reading %s...\n", file);
        views.push_back(imread(file, CV_LOAD_IMAGE_COLOR));
        free(file);
    }

    // Process Images
    unsigned char **views_data = (unsigned char **) malloc(sizeof(unsigned char **) * num_views);
    for (int v = 0; v < num_views; ++v)
    {
        views_data[v] = views[v].data;
    }
    
    int in_width = views[0].cols;
    int in_height = views[0].rows;
    int elem_size = views[0].elemSize();
     
    Mat output = Mat::zeros(out_height, out_width, CV_8UC(3));
    unsigned char* out_data = output.data;
    
    d_mux_multiview( views_data, out_data, num_views, angle, in_width, in_height, out_width, out_height, elem_size);
    
    // Display Images

    namedWindow("Display");
    imshow("Display", views[0]);
	while( 1 )
    {
        char key = waitKey(0);
        int ikey = atoi(&key);
        if (ikey != 0 && key != '0')
        {
            imshow("Display", views[ikey-1]);
        }
        if (key == 'o')
            imshow("Display", output);
    }

    free(views_data);
    
	return 0;

}
