import numpy as np
import cv2
from matplotlib import pyplot as plt
import edge_detect as edge

# define constants
M = 10  # number of columns
N = 8  # number of rows
T = 0.01  # Threshold

# start video capture
cap = cv2.VideoCapture(0)
if not(cap.isOpened()):
    cap.open()

# initialize last frame and edges of last frame
last_frame = np.zeros((480, 640), np.uint8)
last_frame_edge = edge.edge_detect(last_frame)[0]

# motion detection algorithm
def detect_motion(frame, last_frame):
    height, width = np.shape(frame)
    d = last_frame - frame
    motion = np.zeros(((N, M), np.uint8))
    for i in range(M):
        for j in range(N):
            x = 0
            for i_prime in range(height/M):
                for j_prime in range(width/N):
                    if d[i*M+i_prime, j*N+j_prime] != 0:
                        x += 1
            if M*N*x/(height*width*1.0) > T:
                motion[i, j] = 1
    return motion

# main loop
while(cap.isOpened()):
    ret, frame = cap.read()
    frame_edge = edge.edge_detect(frame)[0]
    # cv2.imshow('normal', frame)  # normal video
    cv2.imshow('edge', frame_edge)  # edge detection video
    motion = detect_motion(frame_edge, last_frame_edge)
    cv2.imshow('motion', cv2.resize(motion, (480, 640)))  # motion detection video

    # update frames
    last_frame = frame
    last_frame_edge = frame_edge

    # listen for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()