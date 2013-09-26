#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cvaux.h>

#include "cuda_utils.h"
#include "getCPUtime.h"

#include "d_io.h"

using namespace cv;
using namespace std;

typedef enum
{
    DISPLAY_PERSP_LEFT,
    DISPLAY_PERSP_RIGHT,
} display_persp_e;

typedef enum 
{
    DISPLAY_MODE_SBS,
    DISPLAY_MODE_DISPARITY,
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
    ///////////////////// 
    // LOAD PARAMETERS //
    /////////////////////

    //printDeviceInfo();
    if (argc != 16) 
    {
        printf("Place images in img subdir: \n");
        printf("then input file names directly w/o dir extension \n");
        printf("Usage: ./program [video path] [num views] [angle] [out width] [out height]\n");
        return -1;
    } 
    
    printf("=======================================\n");
    printf("== STEREO TO MULTIVIEW VIDEO PROCESS ==\n");
    printf("=======================================\n\n");

    char* file_vid = argv[1]; 
    
    ////////////////// 
    // FILE PARSING //
    //////////////////

    const char* path_vid = "./vid/";
    char *fullpath_vid = (char *) malloc(snprintf(NULL, 0, "%s%s", path_vid, file_vid) + 1); 
    sprintf(fullpath_vid, "%s%s", path_vid, file_vid);
    printf("Reading %s...\n", fullpath_vid);
    
    ////////////////// 
    // OPEN CAPTURE //
    //////////////////

    // Play in loop
    VideoCapture vc_sbs(fullpath_vid);
    if (!vc_sbs.isOpened())
    {
        printf("Video cannot be read!\nAborting...\n");
        return -1;
    }
    Mat mat_sbs;
    vc_sbs >> mat_sbs;
    int num_frames = vc_sbs.get(CV_CAP_PROP_FRAME_COUNT);
    int num_rows = vc_sbs.get(CV_CAP_PROP_FRAME_HEIGHT);
    int num_cols_sbs =  vc_sbs.get(CV_CAP_PROP_FRAME_WIDTH);
    int num_cols = num_cols_sbs / 2;
    int elem_sz = mat_sbs.elemSize();

    printf("Video Frame Count:       %d\n", num_frames); 
    printf("Input Width (SBS):       %d\n", num_cols_sbs); 
    printf("Input Width (Single):    %d\n", num_cols); 
    printf("Input Height:            %d\n", num_rows); 
    
    int num_views = atoi(argv[2]);
    float angle = atof(argv[3]);
    int num_cols_out = atoi(argv[4]);
    int num_rows_out = atoi(argv[5]);
    int num_disp = atoi(argv[6]);
    int zero_disp = atoi(argv[7]);
    float ad_coeff = atof(argv[8]);
    float census_coeff = atof(argv[9]);
    float ucd = atof(argv[10]);
    float lcd = atof(argv[11]);
    int usd = atoi(argv[12]);
    int lsd = atoi(argv[13]);
    int thresh_s = atoi(argv[14]);
    float thresh_h = atof(argv[15]);


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
    printf("Threshold S:             %d\n", thresh_s);
    printf("Threshold H:             %f\n", thresh_h);
    printf("\n");
    
    int display_mode = DISPLAY_MODE_INTERLACED;
    int display_persp = 0;
    int paused = false;
    
    unsigned char* data_sbs = mat_sbs.data;
            
    Mat mat_disp_l = Mat::zeros(num_rows, num_cols, CV_32F);
    Mat mat_disp_r = Mat::zeros(num_rows, num_cols, CV_32F);
    
    float* data_disp_l = (float*) mat_disp_l.data;
    float* data_disp_r = (float*) mat_disp_r.data;
    
    Mat mat_interlaced = Mat::zeros(num_rows_out, num_cols_out, CV_8UC(3));
    unsigned char* data_interlaced = mat_interlaced.data;
    
    namedWindow("Display");
    for (;;)
    {
        if (paused == false)
        {
            bool read_img = vc_sbs.read(mat_sbs);
            if (read_img == false)
            {
                vc_sbs.set(CV_CAP_PROP_POS_MSEC, 0);
                continue;
            }
        
            // Process SBS image
            double startTime, endTime;
            startTime = getCPUTime();
            adcensus_stm(data_sbs, data_disp_l, data_disp_r, data_interlaced, num_rows, num_cols_sbs, num_cols, num_rows_out, num_cols_out, elem_sz, num_views, angle, num_disp, zero_disp, ad_coeff, census_coeff, ucd, lcd, usd, lsd, thresh_s, thresh_h);
            endTime = getCPUTime();

            fprintf( stderr, "CPU time used = %1f\n", (endTime - startTime));
        
            normalize(mat_disp_l, mat_disp_l, 0, 1, CV_MINMAX);
            normalize(mat_disp_r, mat_disp_r, 0, 1, CV_MINMAX);
        }

        char key = waitKey(33);

        // Handle Keys
        switch (key)
        {
            case '1':
                display_mode = DISPLAY_MODE_SBS;
                break;
            case '2':
                display_mode = DISPLAY_MODE_DISPARITY;
                printf("Showing Disparity\n");
                break;
            case '3':
                display_mode = DISPLAY_MODE_INTERLACED;
                printf("Showing Interlaced\n");
                break;
            case ']':
                if (display_mode == DISPLAY_MODE_DISPARITY)
                    display_persp = DISPLAY_PERSP_RIGHT;
                break;
            case '[':
                if (display_mode == DISPLAY_MODE_DISPARITY)
                    display_persp = DISPLAY_PERSP_LEFT;
                break;
			case 'q':
				return 0;
			case 'p':
                if (paused == false)
				    paused = true;
                else
                    paused = false;
                break;
            default:
                break;
        }
        
        // Handle Display
        switch (display_mode)
        {
            case DISPLAY_MODE_SBS:
                imshow("Display", mat_sbs);
                break;
            case DISPLAY_MODE_DISPARITY:
                if (display_persp == DISPLAY_PERSP_LEFT)
                    imshow("Display", mat_disp_l);
                else if (display_persp == DISPLAY_PERSP_RIGHT)
                    imshow("Display", mat_disp_r);
                break;
            case DISPLAY_MODE_INTERLACED:
                imshow("Display", mat_interlaced);
                break;
            default:
                break;
        }
    }

    return 0;
}
