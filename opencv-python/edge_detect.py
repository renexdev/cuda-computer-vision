import cv2
from matplotlib import pyplot as plt
from os import listdir, getcwd
from os.path import join, isfile
import datetime as dt

images_path = join(getcwd(), '../images')
images = [f for f in listdir(images_path) if isfile(join(images_path, f))]

def edge_detect(image_path, display=False):
    img = cv2.imread(image_path, 0)
    start_time = dt.datetime.now()
    edges = cv2.Canny(img, 100, 200)
    end_time = dt.datetime.now()
    time = (end_time-start_time).microseconds

    if display:
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    return edges, time