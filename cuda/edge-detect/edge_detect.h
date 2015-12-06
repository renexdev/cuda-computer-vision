#ifndef EDGE_DETECT_H
#define EDGE_DETECT_H

void edge_detect(int *edges_out, int *gx_out, int *gy_out, int *image, int image_width, int image_height, int high_threshold, int low_threshold);

void gradient_magnitude_angle_thresholding_and_suppresion(double *dev_magnitude, double *dev_angle, int sobel_out_width, int sobel_out_height, int *dev_gx, int *dev_gy, int *dev_edges, int *edges_out, int high_threshold, int low_threshold, dim3 grid_size, dim3 block_size);

void serial_thresholding_and_suppression(int *output, int input_width, int input_height, int *g_x, int *g_y, int high_threshold, int low_threshold);

#endif
