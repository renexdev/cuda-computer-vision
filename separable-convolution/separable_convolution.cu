#include <stdio.h>
#include <sys/time.h>

__global__ void horizontal_convolve(int *d_out, int *x, int *h, int x_width, int x_height, int h_width, int h_height) {
    const int r = blockIdx.x;
    const int c = threadIdx.x;
    const int i = r * blockDim.x + c;

    int sum = 0;
    for (int j = 0; j < h_width; j++) {
        int p = x_width*r + c - j;
        if (c - j >= 0 && c - j < h_width) {
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
    for (int j = 0; j < x_height; j++) {
        int p = h_width*(r - j) + c;
        if (r - j >= 0 && r - j < h_height) {
            sum += x[j] * h[p];
        }
    }
    d_out[i] = sum;
    __syncthreads();
}

void serial_convolve(int *out, int *x, int *h, int x_width, int x_height, int h_width, int h_height) {
    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);
    for (int m = 0; m < x_height + h_height - 1; m++) {
        for (int n = 0; n < x_width + h_width - 1; n++) {
            int sum = 0;
            for (int i = 0; i < x_height; i++) {
                for (int j = 0; j < x_width; j++) {
                    if (m - i >= 0 && m - i < h_height && n - j >= 0 && n - j < h_width) {
                        sum += x[i * x_width + j] * h[(m - i) * h_width + n - j];
                    }
                }
            }
            out[m * (x_width + h_width - 1) + n] = sum;
        }
    }
    gettimeofday(&tv2, NULL);
    printf ("Serial convolution execution time: %f seconds\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
}

void separable_convolve() {
    int *dev_horizontal_out, *dev_vertical_out;  // Results of the horizontal and vertical convolutions on the input array
    int *dev_horizontal_filter, *dev_vertical_filter, *dev_x;  // Horizontal filter, vertical filter, and input array
    int output[100];

    int horizontal_filter_width = 5;
    int vertical_filter_height = 5;
    int x_width = 2, x_height = 2;

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
    int x[4] = {1, 2, 3, 4};

    cudaMemcpy(dev_horizontal_filter, horizontal_filter, horizontal_filter_width*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vertical_filter, vertical_filter, vertical_filter_height*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, x_width*x_height*sizeof(int), cudaMemcpyHostToDevice);

    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);
    horizontal_convolve<<<horizontal_convolution_height, horizontal_convolution_width>>>(dev_horizontal_out, dev_x, dev_horizontal_filter, x_width, x_height, horizontal_filter_width, 1);
    //vertical_convolve<<<vertical_convolution_height, vertical_convolution_width>>>(dev_vertical_out, dev_horizontal_out, dev_vertical_filter, horizontal_convolution_width, horizontal_convolution_height, 1, vertical_filter_height);

    //cudaMemcpy(output, dev_vertical_out, vertical_convolution_width*vertical_convolution_height*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, dev_horizontal_out, horizontal_convolution_width*horizontal_convolution_height*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_vertical_out);
    cudaFree(dev_horizontal_out);
    cudaFree(dev_horizontal_filter);
    cudaFree(dev_vertical_filter);
    cudaFree(dev_x);

    for (int i = 0; i < horizontal_convolution_width*horizontal_convolution_height; i++) {
        printf("i %d, output %d\n", i, output[i]);
    }

    gettimeofday(&tv2, NULL);
    printf ("Parallel convolution execution time: %f seconds\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
}

int main() {
    separable_convolve();
    return 0;
}
