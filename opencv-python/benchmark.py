from os import listdir, getcwd
from os.path import join, isfile
import edge_detect as edge

images_path = join(getcwd(), '../images')
images = [f for f in listdir(images_path) if isfile(join(images_path, f))]

for img in images:
    edges, time = edge.edge_detect(join(images_path, img))
    print img+":", time