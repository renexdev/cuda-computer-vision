#include <stdio.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../edge-detect/edge_detect.h"
#include "../motion-tracking/motion_tracking.h"
#include "../separable-convolution/separable_convolution.h"
#include "../helper/helper_cuda.h"

using namespace cv;


int main() {
	// Config constants
	double movement_threshold = 5.0;  // Camera shake tolerance
	int motion_threshold = 4;  // Threshold above which motion is registered
	int edge_detect_high_threshold = 70;
	int edge_detect_low_threshold = 50;
	int horizontal_divisions = 12;
	int vertical_divisions = 10;
	
	// Initialize windows
	namedWindow("Input", WINDOW_NORMAL);
	namedWindow("Difference", WINDOW_NORMAL);
	namedWindow("Edges", WINDOW_NORMAL);
	namedWindow("Motion", WINDOW_NORMAL);
	
	// Initialize video stream
	VideoCapture cap(0);
	Mat temp_image;
	cap >> temp_image;
	
	// Allocate host memory
	// Do it all here so as to prevent new allocations happening during the infinite loop below
	int *edges_1 = (int *)malloc((temp_image.cols + 4) * (temp_image.rows + 4) * sizeof(int));
	int *edges_2 = (int *)malloc((temp_image.cols + 4) * (temp_image.rows + 4) * sizeof(int));
	int *gx_out = (int *)malloc((temp_image.cols + 4) * (temp_image.rows + 4) * sizeof(int));
	int *gy_out = (int *)malloc((temp_image.cols + 4) * (temp_image.rows + 4) * sizeof(int));
	int *x_1 = (int *)malloc(temp_image.cols * temp_image.rows * sizeof(int));
	int *x_2 = (int *)malloc(temp_image.cols * temp_image.rows * sizeof(int));
	int *gaussian_out_1 = (int *)malloc((temp_image.cols + 2) * (temp_image.rows + 2) * sizeof(int));
	int *gaussian_out_2 = (int *)malloc((temp_image.cols + 2) * (temp_image.rows + 2) * sizeof(int));
	int *difference = (int *)malloc((temp_image.cols + 4) * (temp_image.rows + 4) * sizeof(int));
	int *motion_area = (int *)malloc((temp_image.cols + 4) * (temp_image.rows + 4) * sizeof(int));
	
	// Infinite loop; real-time motion detection
	for (;;) {
		// Start time counter for approximating FPS
		struct timeval tv1, tv2;
		gettimeofday(&tv1, NULL);
		
		// Continuously read two frames at a time from external camera
		Mat image_1, image_2;
		Mat frame_1, frame_2;
		cap >> frame_1;
		cap >> frame_2;
		cvtColor(frame_1, image_1, COLOR_BGR2GRAY);
		cvtColor(frame_2, image_2, COLOR_BGR2GRAY);
		int input_width = image_1.cols;
		int input_height = image_1.rows;
		for (int i = 0; i < input_height; i++) {
			for (int j = 0; j < input_width; j++) {
				x_1[i * image_1.cols + j] = image_1.at<uchar>(i, j);
				x_2[i * image_2.cols + j] = image_2.at<uchar>(i, j);
			}
		}
	
		// Gaussian filter
		int horizontal_filter[3] = {1, 2, 1};
		int vertical_filter[3] = {1, 2, 1};
		int kernel_size = 3;
		double constant_scalar = 1.0/16.0;
		separable_convolve(gaussian_out_1, x_1, image_1.cols, image_1.rows, horizontal_filter, vertical_filter, kernel_size, constant_scalar);
		separable_convolve(gaussian_out_2, x_2, image_2.cols, image_2.rows, horizontal_filter, vertical_filter, kernel_size, constant_scalar);
		int gaussian_out_width = image_1.cols + kernel_size - 1;
		int gaussian_out_height = image_1.rows + kernel_size - 1;
		
		// Edge detection
		int sobel_out_width = gaussian_out_width + kernel_size - 1;
		int sobel_out_height = gaussian_out_height + kernel_size - 1;
		edge_detect(edges_1, gx_out, gy_out, gaussian_out_1, gaussian_out_width, gaussian_out_height, edge_detect_high_threshold, edge_detect_low_threshold);
		edge_detect(edges_2, gx_out, gy_out, gaussian_out_2, gaussian_out_width, gaussian_out_height, edge_detect_high_threshold, edge_detect_low_threshold);
	
		// Motion detect
		motion_detect(motion_area, difference, edges_1, edges_2, sobel_out_width, sobel_out_height, movement_threshold, motion_threshold, horizontal_divisions, vertical_divisions);
		
		// Display output
		for (int i = 0; i < input_height * input_width; i++) {
			// Before imshow() displays the image, it divides all the values in the matrix by 255, because God knows why.
			// The below three lines of code is a hack to counter that idiocy.
			x_1[i] = 255 * x_1[i];
			difference[i] = 255 * difference[i];
			motion_area[i] = 255 * motion_area[i];
			edges_1[i] = 255 * edges_1[i];
		}
		Mat frame_image_1(input_height, input_width, CV_32SC1, x_1);
		Mat edges_image_1(sobel_out_height, sobel_out_width, CV_32SC1, edges_1);
		Mat difference_image(sobel_out_height, sobel_out_width, CV_32SC1, difference);
		Mat motion_area_image(sobel_out_height, sobel_out_width, CV_32SC1, motion_area);
		imshow("Input", frame_image_1);
		imshow("Edges", edges_image_1);
		imshow("Difference", difference_image);
		imshow("Motion", motion_area_image);
		if (waitKey(30) >= 0)
			break;
		
		// FPS approximation
		gettimeofday(&tv2, NULL);
		double time_spent = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
		char title[16];
		sprintf(title, "Input - %0.2f fps", 1/time_spent);
		setWindowTitle("Input", title);
	}
	
	return 0;
}