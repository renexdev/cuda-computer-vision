#include <stdio.h>
#include <sys/time.h>

__global__ void horizontal_convolve(int *d_out, int *x, int *h, int x_width, int x_height, int h_width, int h_height) {
    const int r = blockIdx.x;
    const int c = threadIdx.x;
    const int i = r * blockDim.x + c;

    int sum = 0;
    for (int j = 0; j < h_width; j++) {
        int p = x_width*r + c - j;
        if (c - j >= 0 && c - j < x_width) {
            sum += h[j] * x[p];
        }
    }
    d_out[i] = sum;
    __syncthreads();
}

__global__ void vertical_convolve(int *d_out, int *x, int *h, int x_width, int x_height, int h_width, int h_height) {
    const int r = blockIdx.x;
    const int c = threadIdx.x;
    const int i = r * blockDim.x + c;

    int sum = 0;
    for (int j = 0; j < h_height; j++) {
        int p = x_width*(r - j) + c;
        if (r - j >= 0 && r - j < x_height) {
            sum += h[j] * x[p];
        }
    }
    d_out[i] = sum;
    __syncthreads();
}

double serial_separable_convolve(int *out, int *x, int *horizontal_filter, int *vertical_filter, int x_width, int x_height, int horizontal_filter_width, int vertical_filter_height) {
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    int *horizontal_out = (int *)malloc((x_width + horizontal_filter_width - 1) * x_height * sizeof(int));
    for (int m = 0; m < x_height; m++) {
        for (int n = 0; n < x_width + horizontal_filter_width - 1; n++) {
            int h_sum = 0;
            for (int j = 0; j < horizontal_filter_width; j++) {
                if (n - j >= 0 && n - j < x_width) {
                    h_sum += x[m * x_width + n - j] * horizontal_filter[j];
                }
            }
            horizontal_out[m * (x_width + horizontal_filter_width - 1) + n] = h_sum;
        }
    }
    for (int v_m = 0; v_m < x_height + vertical_filter_height - 1; v_m++) {
        for (int v_n = 0; v_n < x_width + horizontal_filter_width - 1; v_n++) {
            int v_sum = 0;
            for (int i = 0; i < vertical_filter_height; i++) {
                if (v_m - i >= 0 && v_m - i < x_height) {
                    v_sum += horizontal_out[(v_m - i) * (x_width + horizontal_filter_width - 1) + v_n] * vertical_filter[i];
                }
            }
            out[v_m * (x_width + horizontal_filter_width - 1) + v_n] = v_sum;
        }
    }
    gettimeofday(&tv2, NULL);
    double time_spent = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
    printf ("Serial separated convolution execution time: %f seconds\n", time_spent);
    return time_spent;
}

double serial_convolve(int *out, int *x, int *h, int x_width, int x_height, int h_width, int h_height) {
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    for (int m = 0; m < x_height + h_height - 1; m++) {
        for (int n = 0; n < x_width + h_width - 1; n++) {
            int sum = 0;
            for (int i = 0; i < h_height; i++) {
                for (int j = 0; j < h_width; j++) {
                    if (m - i >= 0 && m - i < x_height && n - j >= 0 && n - j < x_width) {
                        sum += h[i * h_width + j] * x[(m - i) * x_width + n - j];
                    }
                }
            }
            out[m * (x_width + h_width - 1) + n] = sum;
        }
    }
    gettimeofday(&tv2, NULL);
    double time_spent = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
    printf ("Serial naive convolution execution time: %f seconds\n", time_spent);
    return time_spent;
}

void separable_convolve() {
    // Specify lengths of filters and input
    int horizontal_filter_width = 5;
    int vertical_filter_height = 5;
    int x_width = 9000, x_height = 10000;

    // Allocate space for host and device arrays
    int *dev_horizontal_out, *dev_vertical_out;  // Results of the horizontal and vertical convolutions on the input array
    int *dev_horizontal_filter, *dev_vertical_filter, *dev_x;  // Horizontal filter, vertical filter, and input array
    //int *output = (int *)malloc((x_width * x_height + horizontal_filter_width + vertical_filter_height) * sizeof(int));
    //int *serial_output = (int *)malloc((x_width * x_height + horizontal_filter_width + vertical_filter_height) * sizeof(int));
    static int output[100000000], serial_output[100000000];

    // Horizontal filter, followed by vertical filter
    int horizontal_convolution_width = x_width + horizontal_filter_width - 1;
    int horizontal_convolution_height = x_height;
    int vertical_convolution_width = horizontal_convolution_width;
    int vertical_convolution_height = horizontal_convolution_height + vertical_filter_height - 1;

    // Allocate space for horizontal result, vertical result, horizontal filter, vertical filter, and input
    cudaMalloc(&dev_horizontal_out, horizontal_convolution_width*horizontal_convolution_height*sizeof(int));
    cudaMalloc(&dev_vertical_out, vertical_convolution_width*vertical_convolution_height*sizeof(int));
    cudaMalloc(&dev_horizontal_filter, horizontal_filter_width*sizeof(int));
    cudaMalloc(&dev_vertical_filter, vertical_filter_height*sizeof(int));
    cudaMalloc(&dev_x, x_width*x_height*sizeof(int));

    // Load host data
    int horizontal_filter[5] = {1, 2, 3, 4, 5};
    int vertical_filter[5] = {6, 7, 8, 9, 10};
    int *x = (int *)malloc(x_width * x_height * sizeof(int));
    srand(time(NULL));
    for (int k = 0; k < x_width * x_height; k++) {
        x[k] = (int)(rand()*255);
    }

    // Copy host arrays to device
    cudaMemcpy(dev_horizontal_filter, horizontal_filter, horizontal_filter_width*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vertical_filter, vertical_filter, vertical_filter_height*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, x_width*x_height*sizeof(int), cudaMemcpyHostToDevice);

    // Start with a serial convolution for comparison
    // double serial_computation_time = serial_convolve(serial_output, x, filter, x_width, x_height, horizontal_filter_width, vertical_filter_height);
    double serial_computation_time = serial_separable_convolve(serial_output, x, horizontal_filter, vertical_filter, x_width, x_height, horizontal_filter_width, vertical_filter_height);

    // Start a timer and do the two convolutions
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    horizontal_convolve<<<horizontal_convolution_height, horizontal_convolution_width>>>(dev_horizontal_out, dev_x, dev_horizontal_filter, x_width, x_height, horizontal_filter_width, 1);
    vertical_convolve<<<vertical_convolution_height, vertical_convolution_width>>>(dev_vertical_out, dev_horizontal_out, dev_vertical_filter, horizontal_convolution_width, horizontal_convolution_height, 1, vertical_filter_height);

    // Copy result data from device to host
    cudaMemcpy(output, dev_vertical_out, vertical_convolution_width*vertical_convolution_height*sizeof(int), cudaMemcpyDeviceToHost);

    // Responsible programmer
    cudaFree(dev_vertical_out);
    cudaFree(dev_horizontal_out);
    cudaFree(dev_horizontal_filter);
    cudaFree(dev_vertical_filter);
    cudaFree(dev_x);

    // Parallel computation time
    gettimeofday(&tv2, NULL);
    double parallel_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
    printf("Parallel convolution execution time: %f seconds\n", parallel_computation_time);

    // Error checking
    for (int i = 0; i < vertical_convolution_width*vertical_convolution_height; i++) {
        if (serial_output[i] != output[i]) {
            // printf("Error! serial and parallel computation results are inconsistent: %d, %d\n", serial_output[i], output[i]);
        }
    }
    printf("Estimated parallelization speedup: %f\n", serial_computation_time/parallel_computation_time);
}

int main() {
    separable_convolve();
    return 0;
}