from os import listdir, getcwd
from os.path import join, isfile
import edge_detect as edge
import cv2

images_path = join(getcwd(), '../images')
images = [f for f in listdir(images_path) if isfile(join(images_path, f))]

for img_path in images:
    img = cv2.imread(join(images_path, img_path), 0)
    edges, time = edge.edge_detect(img)
    print img_path+":", time