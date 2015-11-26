#include <stdio.h>
#include <sys/time.h>


__global__ void difference_filter(int *dev_out, int *edges_1, int *edges_2, int width, int height, int threshold) {
    // Note: width should correspond to width of dev_out, edges_1, and edges_2; same for height
    const int r = blockIdx.x;
    const int c = threadIdx.x;
    const int i = r * blockDim.x + c;

    // Set it to 0 initially
    dev_out[i] = 0;
    if (edges_1[i] != edges_2[i]) {
        // Set to 1 if there is a pixel mismatch
        dev_out[i] = 1;
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

int main() {
    int x[9] = {1, 1, 1, 0, 0, 0, 0, 0, 0};
    int y[9] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
    int width = 3, height = 3;
    int threshold = 3;

    // Allocate space on device
    int *dev_edges_1, *dev_edges_2, *dev_out;
    cudaMalloc(&dev_out, width*height*sizeof(int));
    cudaMalloc(&dev_edges_1, width*height*sizeof(int));
    cudaMalloc(&dev_edges_2, width*height*sizeof(int));

    // Copy host arrays to device
    cudaMemcpy(dev_edges_1, x, width*height*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_edges_2, y, width*height*sizeof(int), cudaMemcpyHostToDevice);

    difference_filter<<<height, width>>>(dev_out, dev_edges_1, dev_edges_2, width, height, threshold);

    // Copy device result to host
    static int output[10000];
    cudaMemcpy(output, dev_out, width*height*sizeof(int), cudaMemcpyDeviceToHost);

    // Responsible programmer
    cudaFree(dev_out);
    cudaFree(dev_edges_1);
    cudaFree(dev_edges_2);

    for (int i = 0; i < width*height; i++) {
        printf("i %d\n", output[i]);
    }
}
