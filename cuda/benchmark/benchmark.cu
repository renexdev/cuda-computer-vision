#include <stdio.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../separable-convolution/separable_convolution.h"

using namespace cv;


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
	
    printf("==========SERIAL NAIVE VS SEPARABLE CONVOLUTION COMPARISON==========\n");
    compare_naive_separable_convolution(images, 9);
    printf("==========CPU VS GPU SEPARABLE CONVOLUTION SPEEDUP==========\n");
    compare_separable_convolution_speedup(images, 9);

    return 0;
}
