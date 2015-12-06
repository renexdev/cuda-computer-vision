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
#include "../helper/helper_cuda.h"

using namespace cv;

#define TX 8
#define TY 8


void serial_spatial_difference_density_map(double *density_map, int *difference, int width, int height, int horizontal_divisions, int vertical_divisions) {
	int horizontal_block_size = width/horizontal_divisions;
	int vertical_block_size = height/vertical_divisions;
	int block_size = horizontal_block_size * vertical_block_size;
	
	const int scaling_factor = 1000;  // Used to linearly scale density map to units millipixels/pixels^2 (if that makes any sense?)
	
	for (int block_x_index = 0; block_x_index < horizontal_divisions - 1; block_x_index++) {
		for (int block_y_index = 0; block_y_index < vertical_divisions - 1; block_y_index++) {
			int num_differences = 0;
			for (int x = block_x_index * horizontal_block_size; x < (block_x_index + 1) * horizontal_block_size; x++) {
				for (int y = block_y_index * vertical_block_size; y < (block_y_index + 1) * vertical_block_size; y++) {
					if (x > 0 && y > 0 && x < width && y < height && difference[y * width + x] == 255) {
						num_differences++;
					}
				}
			}
			density_map[block_y_index * horizontal_divisions + block_x_index] = scaling_factor*num_differences/(double)block_size;
		}
	}
}

__global__ void spatial_difference_density_map(double *density_map, int *difference, int width, int height, int horizontal_divisions, int vertical_divisions) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int i = r * width + c;
	
	int horizontal_block_size = width/horizontal_divisions;
	int vertical_block_size = height/vertical_divisions;
	int block_size = horizontal_block_size * vertical_block_size;
	
	const int scaling_factor = 1000;
	if (difference[i] != 0) {
		density_map[(int)(vertical_divisions*r/(double)height) * horizontal_divisions + (int)(horizontal_divisions*c/(double)width)] += scaling_factor/(double)block_size;
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
			for (int x = c * horizontal_block_size; x < (c + 1) * horizontal_block_size; x++) {
				for (int y = r * vertical_block_size; y < (r + 1) * vertical_block_size; y++) {
					motion_area[y*width + x] = 255;
				}
			}
		}
	}
}

__global__ void difference_filter(int *dev_out, int *edges_1, int *edges_2, int width, int height, int threshold) {
    // Note: width should correspond to width of dev_out, edges_1, and edges_2; same for height
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int i = r * width + c;

    // Set it to 0 initially
    dev_out[i] = 0;
    int crop_size = 7;
    if (r > crop_size && c > crop_size && r < height - crop_size && c < width - crop_size && edges_1[i] != edges_2[i]) {
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
    __syncthreads();
}

void serial_difference_filter(int *difference, int *edges_1, int *edges_2, int width, int height, int threshold) {
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
}

void motion_detect(int *motion_area, int *difference, int *edges_1, int *edges_2, int width, int height, int movement_threshold, int motion_threshold, int horizontal_divisions, int vertical_divisions) {
	// Note: movement_threshold refers to the pixel apron around which the difference filter attempts to look for differences.
	// Higher movement_threshold == more leniency in how much camera shake is tolerated
	// motion_threshold refers to the minimum spatial pixel difference density required for a particular segment of the difference to be registered as motion.
	// Lower motion_threshold == more sensitive in picking up motion
	
    // Allocate space on device
    int *dev_edges_1, *dev_edges_2, *dev_difference;
    checkCudaErrors(cudaMalloc(&dev_edges_1, width*height*sizeof(int)));
    checkCudaErrors(cudaMalloc(&dev_edges_2, width*height*sizeof(int)));
    checkCudaErrors(cudaMalloc(&dev_difference, width*height*sizeof(int)));
    
    double *density_map = (double *)calloc(horizontal_divisions * vertical_divisions, sizeof(double));
    double *dev_density;
    checkCudaErrors(cudaMalloc(&dev_density, horizontal_divisions*vertical_divisions*sizeof(double)));
    checkCudaErrors(cudaMemcpy(dev_density, density_map, horizontal_divisions * vertical_divisions * sizeof(double), cudaMemcpyHostToDevice));

    // Copy host arrays to device
    checkCudaErrors(cudaMemcpy(dev_edges_1, edges_1, width*height*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_edges_2, edges_2, width*height*sizeof(int), cudaMemcpyHostToDevice));

//    int *serial_output = (int *)malloc(width * height * sizeof(int));
//    serial_difference_filter(difference, edges_1, edges_2, width, height, movement_threshold);

    // Initialize grid
	dim3 block_size(TX, TY);
	int bx = width/block_size.x;
	int by = height/block_size.y;
	dim3 grid_size = dim3(bx, by);

	// Difference filter
    difference_filter<<<grid_size, block_size>>>(dev_difference, dev_edges_1, dev_edges_2, width, height, movement_threshold);
    checkCudaErrors(cudaMemcpy(difference, dev_difference, width * height * sizeof(int), cudaMemcpyDeviceToHost));

    // Determine spatial density map
	spatial_difference_density_map<<<grid_size, block_size>>>(dev_density, dev_difference, width, height, horizontal_divisions, vertical_divisions);
	checkCudaErrors(cudaMemcpy(density_map, dev_density, horizontal_divisions * vertical_divisions * sizeof(double), cudaMemcpyDeviceToHost));
//	serial_spatial_difference_density_map(density_map, difference, width, height, horizontal_divisions, vertical_divisions);
	
	// Estimate motion area
	motion_area_estimate(motion_area, density_map, width, height, horizontal_divisions, vertical_divisions, motion_threshold);
	
	// Responsible programmer
	checkCudaErrors(cudaFree(dev_density));
	checkCudaErrors(cudaFree(dev_difference));
	checkCudaErrors(cudaFree(dev_edges_1));
	checkCudaErrors(cudaFree(dev_edges_2));
	
	// Free allocated host memory
	free(density_map);
}

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
//		printf("=====GAUSSIAN FILTER=====\n");
		int horizontal_filter[3] = {1, 2, 1};
		int vertical_filter[3] = {1, 2, 1};
		int kernel_size = 3;
		double constant_scalar = 1.0/16.0;
		separable_convolve(gaussian_out_1, x_1, image_1.cols, image_1.rows, horizontal_filter, vertical_filter, kernel_size, constant_scalar);
		separable_convolve(gaussian_out_2, x_2, image_2.cols, image_2.rows, horizontal_filter, vertical_filter, kernel_size, constant_scalar);
		int gaussian_out_width = image_1.cols + kernel_size - 1;
		int gaussian_out_height = image_1.rows + kernel_size - 1;
		
		// Edge detect
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
