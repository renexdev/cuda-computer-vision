import numpy as np
import cv2
from matplotlib import pyplot as plt
import edge_detect as edge

# define constants
M = 6  # number of columns
N = 10  # number of rows
T = 0.01  # Threshold
B = 12  # movement tolerance


# start video capture
cap = cv2.VideoCapture(0)
if not(cap.isOpened()):
    cap.open()

# create gaussian filter size 3x3
kernel = np.ones((3, 3), np.float32)/9

# initialize last frame and edges of last frame
last_frame_edge = np.zeros((480, 640), np.uint8)


def difference(frame1, frame2):
    height, width = np.shape(frame1)
    diff = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            if frame1[i, j] != frame2[i, j]:
                diff[i, j] = 255
                for i_prime in range(-B, B):
                    for j_prime in range(-B, B):
                        if i+i_prime < height and i+i_prime >= 0 and j+j_prime < width and j+j_prime >= 0:
                            if frame1[i+i_prime, j+j_prime] == frame2[i+i_prime, j+j_prime]:
                                diff[i+i_prime, j+j_prime] = 0
    return diff


# motion detection algorithm
def detect_motion(edge, last_edge):
    height, width, depth = np.shape(frame)
    d = difference(edge, last_edge)
    motion = np.zeros((M, N), np.uint8)
    for i in range(M):
        for j in range(N):
            x = 0
            for i_prime in range(height/M):
                for j_prime in range(width/N):
                    if d[i*height/M+i_prime, j*width/N+j_prime] != 0:
                        x += 1
            if M*N*x/(height*width*1.0) > T:
                motion[i, j] = 255
    return motion

# main loop
while cap.isOpened():
    ret, frame = cap.read()
    frame_filt = cv2.filter2D(frame, -1, kernel)
    frame_edge = edge.edge_detect(frame_filt)

    # cv2.imshow('normal', frame)  # normal video
    cv2.imshow('edge', frame_edge)  # edge detection video)
    motion = detect_motion(frame_edge, last_frame_edge)
    cv2.imshow('motion', cv2.resize(motion, (640, 480)))  # motion detection video
    # cv2.imshow('difference', difference(frame_edge, last_frame_edge))

    # update frames
    last_frame_edge = frame_edge

    # listen for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()