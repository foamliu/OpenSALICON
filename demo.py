from Salicon import Salicon
import cv2 as cv
import numpy as np
import argparse

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to image file")
    args = vars(ap.parse_args())
    image_path = args["image"]
    if image_path is None:
        image_path = 'face.jpg'

    sal = Salicon()
    image = cv.imread(image_path)
    height, width = image.shape[:2]
    map = sal.compute_saliency(image_path)
    # map contains saliency map in double format.

    print('np.max(map): ' + str(np.max(map)))
    print('np.min(map): ' + str(np.min(map)))

    map = np.reshape(map, (height, width, 1))
    image = image * map
    image = image.astype(np.uint8)

    map = map * 255.
    map = np.min(255, map)
    map = np.max(0, map)
    map = map.astype(np.uint8)

    cv.imwrite('map.png', map)
    cv.imwrite('merged.png', image)


