#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cvaux.h>
#include "cuda_utils.h"
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
    ///////////////////// 
    // LOAD PARAMETERS //
    /////////////////////

    //printDeviceInfo();
    if (argc != 6) 
    {
        printf("Place images in img subdir: \n");
        printf("then input file names directly w/o dir extension \n");
        printf("Usage: ./program [video path] [num views] [angle] [out width] [out height]\n");
        return -1;
    } 

    char* file_vid = argv[1]; 
	int num_views = atoi(argv[2]);
	float angle = atof(argv[3]);
	int num_cols_out = atoi(argv[4]);
	int num_rows_out = atoi(argv[5]);
	int num_disp = 96;
	int zero_disp = 32;
	float ad_coeff = 10.0f;
	float census_coeff = 30.0f;
	float ucd = 17.0f;
	float lcd = 34.0f;
	int usd = 20;
	int lsd = 6;

	printf("/////////////////////////\n");
	printf("// STEREO TO MULTIVIEW //\n");
	printf("/////////////////////////\n\n");

	printf("Number of Views: %d\n", num_views);
	printf("Angle of Attenuator: %f\n", angle);
	printf("Output Width: %d\n", num_cols_out); 
	printf("Output Height: %d\n", num_rows_out); 
	
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

	printf("Video Frame Count: %d\n", num_frames); 
	printf("Input Width (Side by Side): %d\n", num_cols_sbs); 
	printf("Input Width (Single): %d\n", num_cols); 
	printf("Input Height: %d\n", num_rows); 
	
	int display_mode = DISPLAY_MODE_SBS;
    int display_persp = 0;
	int view_num = 0;
    
    namedWindow("Display");
    for (;;)
    {
        bool read_img = vc_sbs.read(mat_sbs);
        if (read_img == false)
        {
            vc_sbs.set(CV_CAP_PROP_POS_MSEC, 0);
            continue;
        }
        
		// Process SBS image

		unsigned char* data_sbs = mat_sbs.data;
		
		Mat mat_disp_l = Mat::zeros(num_rows, num_cols, CV_32F);
		Mat mat_disp_r = Mat::zeros(num_rows, num_cols, CV_32F);

		float* data_disp_l = (float*) mat_disp_l.data;
		float* data_disp_r = (float*) mat_disp_r.data;
		
		std::vector<Mat> mat_views;
		for (int v = 0; v < num_views; ++v)
			mat_views.push_back(Mat::zeros(num_rows, num_cols, CV_8UC(3)));

		unsigned char **data_views = (unsigned char **) malloc(sizeof(unsigned char *) * num_views);
		
		for (int v = 0; v < num_views; ++v)
			data_views[v] = mat_views[v].data;

		Mat mat_interlaced = Mat::zeros(num_rows_out, num_cols_out, CV_8UC(3));

		unsigned char* data_interlaced = mat_interlaced.data;
		
		adcensus_stm(data_sbs, data_disp_l, data_disp_r, data_views, data_interlaced, num_rows, num_cols_sbs, num_cols, num_rows_out, num_cols_out, elem_sz, num_views, angle, num_disp, zero_disp, ad_coeff, census_coeff, ucd, lcd, usd, lsd);
	
		normalize(mat_disp_l, mat_disp_l, 0, 1, CV_MINMAX);
		normalize(mat_disp_r, mat_disp_r, 0, 1, CV_MINMAX);

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
            	display_mode = DISPLAY_MODE_MULTIVIEW;
				break;
			case '4':
            	display_mode = DISPLAY_MODE_INTERLACED;
				printf("Showing Interlaced\n");
				break;
			case ']':
				if (display_mode == DISPLAY_MODE_MULTIVIEW)
				{
					view_num = min(view_num + 1, num_views - 1);
					printf("Showing View # %d\n", view_num + 1);
				}
				else if (display_mode == DISPLAY_MODE_DISPARITY)
					display_persp = DISPLAY_PERSP_RIGHT;
				break;
			case '[':
				if (display_mode == DISPLAY_MODE_MULTIVIEW)
				{
					view_num = max(view_num - 1, 0);
					printf("Showing View # %d\n", view_num + 1);
				}
				else if (display_mode == DISPLAY_MODE_DISPARITY)
					display_persp = DISPLAY_PERSP_LEFT;
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
			case DISPLAY_MODE_MULTIVIEW:
				imshow("Display", mat_views[view_num]);
				break;
			case DISPLAY_MODE_INTERLACED:
				imshow("Display", mat_interlaced);
				break;
			default:
				break;
		}

		free(data_views);
	}

	return 0;
}
