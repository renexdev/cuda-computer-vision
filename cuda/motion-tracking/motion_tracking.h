#ifndef MOTION_TRACKING_H
#define MOTION_TRACKING_H

void motion_detect(int *motion_area, int *difference, int *edges_1, int *edges_2, int width, int height, int movement_threshold, int motion_threshold, int horizontal_divisions, int vertical_divisions);

void serial_motion_detect(int *motion_area, int *difference, int *edges_1, int *edges_2, int width, int height, int movement_threshold, int motion_threshold, int horizontal_divisions, int vertical_divisions);

#endif
