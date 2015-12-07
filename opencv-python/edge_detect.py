import cv2
from matplotlib import pyplot as plt


def edge_detect(img, display=False):

    edges = cv2.Canny(img, 100, 200)

    if display:
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    return edges