#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../edge-detect/edge_detect.h"
#include "../separable-convolution/separable_convolution.h"

using namespace cv;

#define TX 8
#define TY 8


void spatial_difference_density_map(double *density_map, int *difference, int width, int height, int horizontal_divisions, int vertical_divisions) {
	int horizontal_block_size = width/horizontal_divisions;
	int vertical_block_size = height/vertical_divisions;
	int block_size = horizontal_block_size * vertical_block_size;
	
	const int scaling_factor = 1000; // Used to linearly scale density map to units millipixels/pixels^2 (if that makes any sense?)
	
	for (int block_x_index = 0; block_x_index < horizontal_divisions; block_x_index++) {
		for (int block_y_index = 0; block_y_index < vertical_divisions; block_y_index++) {
			int num_differences = 0;
			for (int x = (block_x_index - 1) * horizontal_block_size; x < block_x_index * horizontal_block_size; x++) {
				for (int y = (block_y_index - 1) * vertical_block_size; y < block_y_index * vertical_block_size; y++) {
					if (x > 0 && y > 0 && x < width && y < height && difference[y * width + x] == 255) {
						num_differences++;
					}
				}
			}
			density_map[block_y_index * horizontal_divisions + block_x_index] = scaling_factor*num_differences/(double)block_size;
		}
	}
}

void motion_area_estimate(int *motion_area, double *density_map, int width, int height, int horizontal_divisions, int vertical_divisions, double threshold) {
	int horizontal_block_size = width/horizontal_divisions;
	int vertical_block_size = height/vertical_divisions;
	
	for (int init_i = 0; init_i < width * height; init_i++) {
		motion_area[init_i] = 0;
	}
	for (int i = 0; i < horizontal_divisions * vertical_divisions; i++) {
		if (density_map[i] >= threshold) {
			int r = i/horizontal_divisions;
			int c = i - r*horizontal_divisions;
			for (int x = (c - 1) * horizontal_block_size; x < c * horizontal_block_size; x++) {
				for (int y = (r - 1) * vertical_block_size; y < r * vertical_block_size; y++) {
					motion_area[y*width + x] = 255;
				}
			}
		}
	}
}

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

    printf("=====DIFFERENCE FILTER=====\n");
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
	namedWindow("Input", WINDOW_NORMAL);
	namedWindow("Difference", WINDOW_NORMAL);
	namedWindow("Motion", WINDOW_NORMAL);
	VideoCapture cap(0);
	for (;;) {
		Mat image_1, image_2;
		Mat frame_1, frame_2;
		cap >> frame_1;
		cap >> frame_2;
		cvtColor(frame_1, image_1, COLOR_BGR2GRAY);
		cvtColor(frame_2, image_2, COLOR_BGR2GRAY);
		int input_width = image_1.cols;
		int input_height = image_1.rows;
		int *x_1 = (int *)malloc(input_width * input_height * sizeof(int));
		int *x_2 = (int *)malloc(input_width * input_height * sizeof(int));
		static int *gaussian_out_1 = (int *)malloc((input_width * input_height + 10) * sizeof(int));
		static int *gaussian_out_2 = (int *)malloc((input_width * input_height + 10) * sizeof(int));
		for (int i = 0; i < input_height; i++) {
			for (int j = 0; j < input_width; j++) {
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
	
		static int *difference = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
		int threshold = 5;
		motion_track(difference, edges_1, edges_2, sobel_out_width, sobel_out_height, threshold);
		
		int horizontal_divisions = 12, vertical_divisions = 10;
		double *density_map = (double *)malloc(horizontal_divisions * vertical_divisions * sizeof(double));
		spatial_difference_density_map(density_map, difference, sobel_out_width, sobel_out_height, horizontal_divisions, vertical_divisions);
		int *motion_area = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
		double motion_threshold = 10.0;
		motion_area_estimate(motion_area, density_map, sobel_out_width, sobel_out_height, horizontal_divisions, vertical_divisions, motion_threshold);
		
		for (int i = 0; i < horizontal_divisions * vertical_divisions; i++) {
	//    	printf("%f\n", density_map[i]);
		}
		
		for (int i = 0; i < input_height * input_width; i++) {
			// Before imshow() displays the image, it divides all the values in the matrix by 255, because God knows why.
			// The below three lines of code is a hack to counter that idiocy.
			x_1[i] = 255 * x_1[i];
			difference[i] = 255 * difference[i];
			motion_area[i] = 255 * motion_area[i];
		}
		Mat frame_image_1(input_height, input_width, CV_32SC1, x_1);
		Mat frame_image_2(input_height, input_width, CV_32SC1, x_2);
		Mat edges_image_1(sobel_out_height, sobel_out_width, CV_32SC1, edges_1);
		Mat edges_image_2(sobel_out_height, sobel_out_width, CV_32SC1, edges_2);
		Mat difference_image(sobel_out_height, sobel_out_width, CV_32SC1, difference);
		Mat motion_area_image(sobel_out_height, sobel_out_width, CV_32SC1, motion_area);
		imshow("Input", frame_image_1);
		imshow("Difference", difference_image);
		imshow("Motion", motion_area_image);
		if (waitKey(30) >= 0)
			break;
	}
	
	return 0;
}