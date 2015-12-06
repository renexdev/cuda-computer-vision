#include <stdio.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../separable-convolution/separable_convolution.h"
#include "../motion-tracking/motion_tracking.h"
#include "../edge-detect/edge_detect.h"
#include "../helper/helper_cuda.h"

using namespace cv;

#define TX 16
#define TY 16


void compare_naive_separable_convolution(const char **images, int num_images) {
	// Benchmark serial (CPU) performance of naive convolution versus separated convolution on a fixed filter and randomly-valued input
	for (int image_index = 0; image_index < num_images; image_index++) {
		Mat image = imread(images[image_index], 0);
		int input_width = image.cols;
		int input_height = image.rows;
		int *out = (int *)malloc((input_width + 2) * (input_height + 2) * sizeof(int));
		int h[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
		int h_horizontal[3] = {1, 2, 1};
		int h_vertical[3] = {1, 2, 1};
		int *x = (int *)malloc(input_width * input_height * sizeof(int));
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				x[i * image.cols + j] = image.at<uchar>(i, j);
			}
		}
		struct timeval tv1, tv2;
		gettimeofday(&tv1, NULL);
		serial_naive_convolve(out, x, h, input_width, input_height, 3, 3);
		gettimeofday(&tv2, NULL);
		double naive_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
	
		gettimeofday(&tv1, NULL);
		serial_separable_convolve(out, x, h_horizontal, h_vertical, input_width, input_height, 3, 3, 1.0);
		gettimeofday(&tv2, NULL);
		double separable_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);

		printf("Test image: %s\n", images[image_index]);
		printf("Serial naive convolution execution time: %f\n", naive_computation_time);
		printf("Serial separable convolution execution time: %f\n", separable_computation_time);
	}
}

void compare_separable_convolution_speedup(const char **images, int num_images) {
    // Benchmark parallel speedup of Gaussian filter - serial CPU separated convolution vs. parallel GPU separated convolution
	for (int image_index = 0; image_index < num_images; image_index++) {
		Mat image = imread(images[image_index], 0);
		int input_width = image.cols;
		int input_height = image.rows;
		
		int *out = (int *)malloc((input_width + 2) * (input_height + 2) * sizeof(int));
		int h_horizontal[3] = {1, 2, 1};
		int h_vertical[3] = {1, 2, 1};
		int *x = (int *)malloc(input_width * input_height * sizeof(int));
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				x[i * image.cols + j] = image.at<uchar>(i, j);
			}
		}
		struct timeval tv1, tv2;
		gettimeofday(&tv1, NULL);
		separable_convolve(out, x, input_width, input_height, h_horizontal, h_vertical, 3, 1.0);
		gettimeofday(&tv2, NULL);
		double parallel_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
	
		gettimeofday(&tv1, NULL);
		serial_separable_convolve(out, x, h_horizontal, h_vertical, input_width, input_height, 3, 3, 1.0);
		gettimeofday(&tv2, NULL);
		double serial_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		double estimated_speedup = serial_computation_time/parallel_computation_time;
	
		printf("Test image: %s\n", images[image_index]);
		printf("Parallel separable convolution execution time: %f\n", parallel_computation_time);
		printf("Serial separable convolution execution time: %f\n", serial_computation_time);
		printf("Estimated parallelization speedup: %f\n", estimated_speedup);
	}
}

void non_maximum_suppression_selective_thresholding_speedup(const char **images, int num_images) {
	int high_threshold = 70;
	int low_threshold = 50;
	
	for (int image_index = 0; image_index < num_images; image_index++) {
		Mat raw_image = imread(images[image_index], 0);
		int *image = (int *)malloc(raw_image.cols * raw_image.rows * sizeof(int));
		for (int i = 0; i < raw_image.rows; i++) {
			for (int j = 0; j < raw_image.cols; j++) {
				image[i * raw_image.cols + j] = raw_image.at<uchar>(i, j);
			}
		}
		
		int *gx_out = (int *)malloc((raw_image.cols + 4) * (raw_image.rows + 4) * sizeof(int));
		int *gy_out = (int *)malloc((raw_image.cols + 4) * (raw_image.rows + 4) * sizeof(int));
		int *edges_out = (int *)malloc((raw_image.cols + 4) * (raw_image.rows + 4) * sizeof(int));
		
		int kernel_size = 3;
		int sobel_out_width = raw_image.cols + kernel_size - 1;
		int sobel_out_height = raw_image.rows + kernel_size - 1;
		dim3 block_size(TX, TY);
		int bx = sobel_out_width/block_size.x;
		int by = sobel_out_height/block_size.y;
		dim3 grid_size = dim3(bx, by);
		
		// Horizontal direction
		int gx_horizontal[3] = {1, 0, -1};
		int gx_vertical[3] = {1, 2, 1};
		separable_convolve(gx_out, image, sobel_out_width, sobel_out_height, gx_horizontal, gx_vertical, 3, 1);
		
		// Vertical direction
		int gy_horizontal[3] = {1, 2, 1};
		int gy_vertical[3] = {1, 0, -1};
		separable_convolve(gy_out, image, sobel_out_width, sobel_out_height, gy_horizontal, gy_vertical, 3, 1);
		
		int *dev_edges, *dev_gx, *dev_gy;
		double *dev_magnitude, *dev_angle;
		
		// Allocate GPU memory space for partial derivatives
		checkCudaErrors(cudaMalloc(&dev_magnitude, sobel_out_width * sobel_out_height * sizeof(double)));
		checkCudaErrors(cudaMalloc(&dev_angle, sobel_out_width * sobel_out_height * sizeof(double)));
		checkCudaErrors(cudaMalloc(&dev_edges, sobel_out_width * sobel_out_height * sizeof(int)));
		checkCudaErrors(cudaMalloc(&dev_gx, sobel_out_width * sobel_out_height * sizeof(int)));
		checkCudaErrors(cudaMalloc(&dev_gy, sobel_out_width * sobel_out_height * sizeof(int)));
		checkCudaErrors(cudaMemcpy(dev_gx, gx_out, sobel_out_width * sobel_out_height * sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dev_gy, gy_out, sobel_out_width * sobel_out_height * sizeof(int), cudaMemcpyHostToDevice));
		
		struct timeval tv1, tv2;
		gettimeofday(&tv1, NULL);
		gradient_magnitude_angle_thresholding_and_suppresion(dev_magnitude, dev_angle, sobel_out_width, sobel_out_height, dev_gx, dev_gy, dev_edges, edges_out, high_threshold, low_threshold, grid_size, block_size);
		gettimeofday(&tv2, NULL);
		double parallel_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		
		gettimeofday(&tv1, NULL);
		serial_thresholding_and_suppression(edges_out, sobel_out_width, sobel_out_height, gx_out, gy_out, high_threshold, low_threshold);
		gettimeofday(&tv2, NULL);
		double serial_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		
		checkCudaErrors(cudaFree(dev_edges));
		checkCudaErrors(cudaFree(dev_gx));
		checkCudaErrors(cudaFree(dev_gy));
		checkCudaErrors(cudaFree(dev_magnitude));
		checkCudaErrors(cudaFree(dev_angle));
		
		printf("Test image: %s\n", images[image_index]);
		printf("Parallel non-maximum suppression and selective thresholding execution time: %f seconds\n", parallel_computation_time);
		printf("Serial non-maximum suppression and selective thresholding execution time: %f seconds\n", serial_computation_time);
		printf("Estimated parallelization speedup: %f\n", serial_computation_time/parallel_computation_time);
	}
}

void motion_area_estimation_speedup(const char **images, int num_images) {
	// Uses random matrices for the difference
	double movement_threshold = 5.0;
	int motion_threshold = 4;
	int horizontal_divisions = 5;
	int vertical_divisions = 5;
	
	struct timeval tv1, tv2;
	
	for (int image_index = 0; image_index < num_images; image_index++) {
		Mat raw_image = imread(images[image_index], 0);
		int *motion_area = (int *)malloc(raw_image.cols * raw_image.rows * sizeof(int));
		int *difference = (int *)malloc(raw_image.cols * raw_image.rows * sizeof(int));
		int *edges_1 = (int *)malloc(raw_image.cols * raw_image.rows * sizeof(int));
		int *edges_2 = (int *)malloc(raw_image.cols * raw_image.rows * sizeof(int));
		
		for (int i = 0; i < raw_image.cols * raw_image.rows; i++) {
			edges_1[i] = rand() % 2;
			edges_2[i] = rand() % 2;
		}
		
		gettimeofday(&tv1, NULL);
		motion_detect(motion_area, difference, edges_1, edges_2, raw_image.cols, raw_image.rows, movement_threshold, motion_threshold, horizontal_divisions, vertical_divisions);
		gettimeofday(&tv2, NULL);
		double parallel_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		
		gettimeofday(&tv1, NULL);
		serial_motion_detect(motion_area, difference, edges_1, edges_2, raw_image.cols, raw_image.rows, movement_threshold, motion_threshold, horizontal_divisions, vertical_divisions);
		gettimeofday(&tv2, NULL);
		double serial_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		
		printf("Test image: %s\n", images[image_index]);
		printf("Parallel motion area estimation execution time: %f seconds\n", parallel_computation_time);
		printf("Serial motion area estimation execution time: %f seconds\n", serial_computation_time);
		printf("Estimated parallelization speedup: %f\n", serial_computation_time/parallel_computation_time);
	}
}

int main() {
    const char *images[9];
    images[0] = "../../images/city_100_100.jpg";
    images[1] = "../../images/city_500_500.jpg";
    images[2] = "../../images/city_1000_1000.jpg";
    images[3] = "../../images/city_2000_2000.jpg";
    images[4] = "../../images/city_3000_3000.jpg";
    images[5] = "../../images/city_4000_4000.jpg";
    images[6] = "../../images/city_5000_5000.jpg";
    images[7] = "../../images/city_6000_6000.jpg";
    images[8] = "../../images/city_7500_7500.jpg";
    
    srand(time(NULL));
	
    printf("==========SERIAL NAIVE VS SEPARABLE CONVOLUTION COMPARISON==========\n");
    compare_naive_separable_convolution(images, 9);
    printf("==========CPU VS GPU SEPARABLE CONVOLUTION SPEEDUP==========\n");
    compare_separable_convolution_speedup(images, 9);
    printf("==========CPU VS GPU NON-MAXIMUM SUPPRESSION AND SELECTIVE THRESHOLDING SPEEDUP==========\n");
    non_maximum_suppression_selective_thresholding_speedup(images, 8);
    printf("==========CPU VS GPU MOTION AREA ESTIMATION SPEEDUP==========\n");
    motion_area_estimation_speedup(images, 9);

    return 0;
}
