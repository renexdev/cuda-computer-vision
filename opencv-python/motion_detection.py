import numpy as np
import cv2
from matplotlib import pyplot as plt
import edge_detect as edge

cap = cv2.VideoCapture(0)

if not(cap.isOpened()):
    cap.open()

last_frame = np.zeros((480, 640), np.uint8)
last_frame_edge = edge.edge_detect(last_frame)[0]
while(cap.isOpened()):
    ret, frame = cap.read()
    frame_edge = edge.edge_detect(frame)[0]
    # cv2.imshow('frame', frame)  # normal picture
    cv2.imshow('edge', frame_edge)  # edge detection picture
    cv2.imshow('motion', frame_edge-last_frame_edge)
    last_frame = frame
    last_frame_edge = frame_edge
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()