#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../edge-detect/edge_detect.h"
#include "../separable-convolution/separable_convolution.h"

using namespace cv;

#define TX 8
#define TY 8


__global__ void difference_filter(int *dev_out, int *edges_1, int *edges_2, int width, int height, int threshold) {
    // Note: width should correspond to width of dev_out, edges_1, and edges_2; same for height
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = r * width + c;

    // Set it to 0 initially
    dev_out[i] = 0;
    if (edges_1[i] != edges_2[i]) {
        // Set to 255 if there is a pixel mismatch
        dev_out[i] = 255;
        for (int x_apron = -threshold; x_apron <= threshold; x_apron++) {
            for (int y_apron = -threshold; y_apron <= threshold; y_apron++) {
                // Ensure the requested index is within bounds of image
                if (c + x_apron > 0 && r + y_apron > 0 && c + x_apron < width && r + y_apron < height) {
                    // Check if there is a matching pixel in the apron, within the threshold
                    if (edges_1[(r + y_apron) * width + c + x_apron] == edges_2[i]) {
                        // Set it back to 0 if a corresponding pixel exists within the vicinity of the match
                        dev_out[i] = 0;
                    }
                }
            }
        }
    }
}

double serial_difference_filter(int *difference, int *edges_1, int *edges_2, int width, int height, int threshold) {
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            difference[y * width + x] = 0;
            if (edges_1[y * width + x] != edges_2[y * width + x]) {
                difference[y * width + x] = 255;
                for (int x_apron = -threshold; x_apron <= threshold; x_apron++) {
                    for (int y_apron = -threshold; y_apron <= threshold; y_apron++) {
                        if (x + x_apron > 0 && y + y_apron > 0 && x + x_apron < width && y + y_apron < height) {
                            if (edges_1[(y + y_apron) * width + x + x_apron] == edges_2[y * width + x]) {
                                difference[y * width + x] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    gettimeofday(&tv2, NULL);
    double time_spent = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
    printf("Serial difference filter execution time: %f seconds\n", time_spent);
    return time_spent;
}

void motion_track(int *output, int *edges_1, int *edges_2, int width, int height, int threshold) {
    // Allocate space on device
    int *dev_edges_1, *dev_edges_2, *dev_out;
    cudaMalloc(&dev_out, width*height*sizeof(int));
    cudaMalloc(&dev_edges_1, width*height*sizeof(int));
    cudaMalloc(&dev_edges_2, width*height*sizeof(int));

    // Copy host arrays to device
    cudaMemcpy(dev_edges_1, edges_1, width*height*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_edges_2, edges_2, width*height*sizeof(int), cudaMemcpyHostToDevice);

    // Allocate space on host for output arrays
    int *serial_output = (int *)malloc(width * height * sizeof(int));

    double serial_time_spent = serial_difference_filter(serial_output, edges_1, edges_2, width, height, threshold);

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    // Initialize grid
	dim3 block_size(TX, TY);
	int bx = width/block_size.x;
	int by = height/block_size.y;
	dim3 grid_size = dim3(bx, by);

    difference_filter<<<grid_size, block_size>>>(dev_out, dev_edges_1, dev_edges_2, width, height, threshold);
    cudaMemcpy(output, dev_out, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    gettimeofday(&tv2, NULL);
    double time_spent = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
    printf("Parallel difference filter execution time: %f seconds\n", time_spent);

    // Responsible programmer
    cudaFree(dev_out);
    cudaFree(dev_edges_1);
    cudaFree(dev_edges_2);

    for (int i = 0; i < width*height; i++) {
        if (serial_output[i] != output[i]) {
            // printf("Error! Serial and parallel computation results are inconsistent: %d, %d\n", serial_output[i], output[i]);
        }
    }
    printf("Estimated parallelization speedup: %f\n", serial_time_spent/time_spent);
}

int main() {
	// Load external image into array
	Mat image_1 = imread("../images/lobby_1.jpg", 0);
	Mat image_2 = imread("../images/lobby_2.jpg", 0);
	int *x_1 = (int *)malloc(image_1.cols * image_1.rows * sizeof(int));
	int *x_2 = (int *)malloc(image_2.cols * image_2.rows * sizeof(int));
	static int *gaussian_out_1 = (int *)malloc((image_1.rows * image_1.cols + 10) * sizeof(int));
	static int *gaussian_out_2 = (int *)malloc((image_2.rows * image_2.cols + 10) * sizeof(int));
	for (int i = 0; i < image_1.rows; i++) {
		for (int j = 0; j < image_1.cols; j++) {
			x_1[i * image_1.cols + j] = image_1.at<uchar>(i, j);
			x_2[i * image_2.cols + j] = image_2.at<uchar>(i, j);
		}
	}

	// Gaussian filter
	printf("=====GAUSSIAN FILTER=====\n");
	int horizontal_filter[3] = {1, 2, 1};
	int vertical_filter[3] = {1, 2, 1};
	int kernel_size = 3;
	double constant_scalar = 1.0/16.0;
	separable_convolve(gaussian_out_1, x_1, image_1.cols, image_1.rows, horizontal_filter, vertical_filter, kernel_size, constant_scalar);
	separable_convolve(gaussian_out_2, x_2, image_2.cols, image_2.rows, horizontal_filter, vertical_filter, kernel_size, constant_scalar);
	int gaussian_out_width = image_1.cols + kernel_size - 1;
	int gaussian_out_height = image_1.rows + kernel_size - 1;
	
	// Edge detect
	int high_threshold = 70, low_threshold = 50;
	int sobel_out_width = gaussian_out_width + kernel_size - 1;
	int sobel_out_height = gaussian_out_height + kernel_size - 1;
	static int *edges_1 = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
	static int *edges_2 = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
	edge_detect(edges_1, gaussian_out_1, gaussian_out_width, gaussian_out_height, high_threshold, low_threshold);
	edge_detect(edges_2, gaussian_out_2, gaussian_out_width, gaussian_out_height, high_threshold, low_threshold);

	static int *out = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
	int threshold = 5;
    motion_track(out, edges_1, edges_2, sobel_out_width, sobel_out_height, threshold);
    
    // Write to disk
	Mat edges_image_1(sobel_out_height, sobel_out_width, CV_32SC1, edges_1);
	Mat edges_image_2(sobel_out_height, sobel_out_width, CV_32SC1, edges_2);
	Mat difference_image(sobel_out_height, sobel_out_width, CV_32SC1, out);
	imwrite("edges_1.jpg", edges_image_1);
	imwrite("edges_2.jpg", edges_image_2);
	imwrite("difference.jpg", difference_image);
}
