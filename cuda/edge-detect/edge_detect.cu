#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../separable-convolution/separable_convolution.h"

using namespace cv;

#define TX 16
#define TY 16


__global__ void gradient_magnitude_and_direction(double *dev_magnitude_output, double *dev_angle_output, int input_width, int input_height, int *g_x, int *g_y) {
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = r * (input_width + 2) + c;
	
	dev_magnitude_output[i] = sqrt(pow((double)g_x[i], 2) + pow((double)g_y[i], 2));
	dev_angle_output[i] = atan2((double)g_y[i], (double)g_x[i]);
}

__global__ void thresholding_and_suppression(int *dev_output, double *dev_magnitude, double *dev_angle, int input_width, int input_height, int *g_x, int *g_y, int high_threshold, int low_threshold) {
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = r * (input_width + 2) + c;

    // First, initialize the current pixel to zero (non-edge)
    dev_output[i] = 0;
    // Boundary conditions
    if (r > 1 && c > 1 && r < input_height - 1 && c < input_width - 1) {
        double magnitude = dev_magnitude[i];
        if (magnitude > high_threshold) {
        	// Immediately accept any pixel above the high threshold
        	dev_output[i] = 255;
        	
        	// Non-maximum suppression: determine magnitudes in the surrounding pixel and the gradient direction of the current pixel
        	double magnitude_above = dev_magnitude[(r - 1) * input_width + c];
			double magnitude_below = dev_magnitude[(r + 1) * input_width + c];
			double magnitude_left = dev_magnitude[r * input_width + c - 1];
			double magnitude_right = dev_magnitude[r * input_width + c + 1];
			double magnitude_upper_right = dev_magnitude[(r + 1) * input_width + c + 1];
			double magnitude_upper_left = dev_magnitude[(r + 1) * input_width + c - 1];
			double magnitude_lower_right = dev_magnitude[(r - 1) * input_width + c + 1];
			double magnitude_lower_left = dev_magnitude[(r - 1) * input_width + c - 1];
			double theta = dev_angle[i];
		
			// Check if the current pixel is a ridge pixel, e.g. maximized in the gradient direction
			int vertical_check = (M_PI/3.0 < theta && theta < 2.0*M_PI/3.0) || (-2.0*M_PI/3.0 < theta && theta < -M_PI/3.0);
			int is_vertical_max = vertical_check && magnitude > magnitude_below && magnitude > magnitude_above;
			int horizontal_check = (-M_PI/6.0 < theta && theta < M_PI/6.0) || (-M_PI < theta && theta < -5.0*M_PI/6.0) || (5*M_PI/6.0 < theta && theta < M_PI);
			int is_horizontal_max = horizontal_check && magnitude > magnitude_right && magnitude > magnitude_left;
	        int positive_diagonal_check = (theta > M_PI/6.0 && theta < M_PI/3.0) || (theta < -2.0*M_PI/3.0 && theta > -5.0*M_PI/6.0);
			int is_positive_diagonal_max = positive_diagonal_check && magnitude > magnitude_upper_right && magnitude > magnitude_lower_left;
			int negative_diagonal_check = (theta > 2.0*M_PI/3.0 && theta < 5.0*M_PI/6.0) || (theta < -M_PI/6.0 && theta > -M_PI/3.0);
			int is_negative_diagonal_max = negative_diagonal_check && magnitude > magnitude_lower_right && magnitude > magnitude_upper_left;
		
			// Consider a surrounding apron around the current pixel to catch potentially disconnected pixel nodes
			int apron_size = 2;
			if (is_vertical_max || is_horizontal_max || is_positive_diagonal_max || is_negative_diagonal_max) {
				for (int m = -apron_size; m <= apron_size; m++) {
					for (int n = -apron_size; n <= apron_size; n++) {
						if (r + m > 0 && r + m < input_height && c + n > 0 && c + n < input_width) {
							if (dev_magnitude[(r + m) * input_width + c + n] > low_threshold) {
								dev_output[(r + m) * input_width + c + n] = 255;
							}
						}
					}
				}
			}
		}
    }
}

double serial_thresholding_and_suppression(int *dev_output, int input_width, int input_height, int *g_x, int *g_y, int high_threshold, int low_threshold) {
	struct timeval tv1, tv2;
	gettimeofday(&tv1, NULL);
	for (int r = 0; r < input_height; r++) {
		for (int c = 0; c < input_width; c++) {
			int i = r * input_width + c;
			// First, initialize the current pixel to zero (non-edge)
			dev_output[i] = 0;
			// Boundary conditions
			if (r > 1 && c > 1 && r < input_height - 1 && c < input_width - 1) {
				double magnitude = sqrt(pow((double)g_x[i], 2) + pow((double)g_y[i], 2));
				if (magnitude > high_threshold) {
					// Immediately accept any pixel above the high threshold
					dev_output[i] = 255;
					
					// Non-maximum suppression: determine magnitudes in the surrounding pixel and the gradient direction of the current pixel
					double magnitude_above = sqrt(pow((double)g_x[(r - 1) * input_width + c], 2) + pow((double)g_y[(r - 1) * input_width + c], 2));
					double magnitude_below = sqrt(pow((double)g_x[(r + 1) * input_width + c], 2) + pow((double)g_y[(r + 1) * input_width + c], 2));
					double magnitude_left = sqrt(pow((double)g_x[r * input_width + c - 1], 2) + pow((double)g_y[r * input_width + c - 1], 2));
					double magnitude_right = sqrt(pow((double)g_x[r * input_width + c + 1], 2) + pow((double)g_y[r * input_width + c + 1], 2));
					double magnitude_upper_right = sqrt(pow((double)g_x[(r + 1) * input_width + c + 1], 2) + pow((double)g_y[(r + 1) * input_width + c + 1], 2));
					double magnitude_upper_left = sqrt(pow((double)g_x[(r + 1) * input_width + c - 1], 2) + pow((double)g_y[(r + 1) * input_width + c - 1], 2));
					double magnitude_lower_right = sqrt(pow((double)g_x[(r - 1) * input_width + c + 1], 2) + pow((double)g_y[(r - 1) * input_width + c + 1], 2));
					double magnitude_lower_left = sqrt(pow((double)g_x[(r - 1) * input_width + c - 1], 2) + pow((double)g_y[(r - 1) * input_width + c - 1], 2));
					double theta = atan2((double)g_y[i], (double)g_x[i]);
				
					// Check if the current pixel is a ridge pixel, e.g. maximized in the gradient direction
					int vertical_check = (M_PI/3.0 < theta && theta < 2.0*M_PI/3.0) || (-2.0*M_PI/3.0 < theta && theta < -M_PI/3.0);
					int is_vertical_max = vertical_check && magnitude > magnitude_below && magnitude > magnitude_above;
					int horizontal_check = (-M_PI/6.0 < theta && theta < M_PI/6.0) || (-M_PI < theta && theta < -5.0*M_PI/6.0) || (5*M_PI/6.0 < theta && theta < M_PI);
					int is_horizontal_max = horizontal_check && magnitude > magnitude_right && magnitude > magnitude_left;
					int positive_diagonal_check = (theta > M_PI/6.0 && theta < M_PI/3.0) || (theta < -2.0*M_PI/3.0 && theta > -5.0*M_PI/6.0);
					int is_positive_diagonal_max = positive_diagonal_check && magnitude > magnitude_upper_right && magnitude > magnitude_lower_left;
					int negative_diagonal_check = (theta > 2.0*M_PI/3.0 && theta < 5.0*M_PI/6.0) || (theta < -M_PI/6.0 && theta > -M_PI/3.0);
					int is_negative_diagonal_max = negative_diagonal_check && magnitude > magnitude_lower_right && magnitude > magnitude_upper_left;
				
					// Consider a surrounding apron around the current pixel to catch potentially disconnected pixel nodes
					int apron_size = 2;
					if (is_vertical_max || is_horizontal_max || is_positive_diagonal_max || is_negative_diagonal_max) {
						for (int m = -apron_size; m <= apron_size; m++) {
							for (int n = -apron_size; n <= apron_size; n++) {
								if (r + m > 0 && r + m < input_height && c + n > 0 && c + n < input_width) {
									if (sqrt(pow((double)g_x[(r + m) * input_width + c + n], 2) + pow((double)g_y[(r + m) * input_width + c + n], 2)) > low_threshold) {
										dev_output[(r + m) * input_width + c + n] = 255;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	gettimeofday(&tv2, NULL);
	double time_spent = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
	printf ("Serial thresholding and non-maximum suppression execution time: %f seconds\n", time_spent);
	return time_spent;
}

void edge_detect(int *edges_out, int *image, int image_width, int image_height, int high_threshold, int low_threshold) {
	// Sobel filter
	int kernel_size = 3;
	int sobel_out_width = image_width + kernel_size - 1;
	int sobel_out_height = image_height + kernel_size - 1;
	dim3 block_size(TX, TY);
	int bx = sobel_out_width/block_size.x;
	int by = sobel_out_height/block_size.y;
	dim3 grid_size = dim3(bx, by);
	
	// Horizontal direction
	printf("=====HORIZONTAL PARTIAL DIFFERENTIATION=====\n");
	int gx_horizontal[3] = {1, 0, -1};
	int gx_vertical[3] = {1, 2, 1};
	static int *gx_out = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
	separable_convolve(gx_out, image, sobel_out_width, sobel_out_height, gx_horizontal, gx_vertical, 3, 1);
	
	// Vertical direction
	printf("=====VERTICAL PARTIAL DIFFERENTIATION=====\n");
	int gy_horizontal[3] = {1, 2, 1};
	int gy_vertical[3] = {1, 0, -1};
	static int *gy_out = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
	separable_convolve(gy_out, image, sobel_out_width, sobel_out_height, gy_horizontal, gy_vertical, 3, 1);

	// Magnitude and thresholding
	static double *magnitude = (double *)malloc(sobel_out_width * sobel_out_height * sizeof(double));
	static double *angle = (double *)malloc(sobel_out_width * sobel_out_height * sizeof(double));
	static int *serial_edges = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
	int *dev_edges, *dev_gx, *dev_gy;
	double *dev_magnitude, *dev_angle;
	
	// Allocate GPU memory space for partial derivatives
	cudaMalloc(&dev_magnitude, sobel_out_width * sobel_out_height * sizeof(double));
	cudaMalloc(&dev_angle, sobel_out_width * sobel_out_height * sizeof(double));
	cudaMalloc(&dev_edges, sobel_out_width * sobel_out_height * sizeof(int));
	cudaMalloc(&dev_gx, sobel_out_width * sobel_out_height * sizeof(int));
	cudaMalloc(&dev_gy, sobel_out_width * sobel_out_height * sizeof(int));
	cudaMemcpy(dev_gx, gx_out, sobel_out_width * sobel_out_height * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gy, gy_out, sobel_out_width * sobel_out_height * sizeof(int), cudaMemcpyHostToDevice);
	
	// Serial comparison
	printf("=====THRESHOLDING AND NON-MAXIMUM SUPPRESSION=====\n");
	double serial_computation_time = serial_thresholding_and_suppression(serial_edges, sobel_out_width, sobel_out_height, gx_out, gy_out, high_threshold, low_threshold);
	
	// Parallelization
	struct timeval tv1, tv2;
	gettimeofday(&tv1, NULL);
	gradient_magnitude_and_direction<<<grid_size, block_size>>>(dev_magnitude, dev_angle, sobel_out_width, sobel_out_height, dev_gx, dev_gy);
	thresholding_and_suppression<<<grid_size, block_size>>>(dev_edges, dev_magnitude, dev_angle, sobel_out_width, sobel_out_height, dev_gx, dev_gy, high_threshold, low_threshold);
	cudaMemcpy(edges_out, dev_edges, sobel_out_width * sobel_out_height * sizeof(int), cudaMemcpyDeviceToHost);
	gettimeofday(&tv2, NULL);
	double parallel_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
	printf("Parallel thresholding and non-maximum suppression execution time: %f seconds\n", parallel_computation_time);
	printf("Estimated parallelization speedup: %f\n", serial_computation_time/parallel_computation_time);

	// Free GPU memory
	cudaFree(dev_edges);
	cudaFree(dev_gx);
	cudaFree(dev_gy);
}