from Salicon import Salicon
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    sal = Salicon()
    image = cv.imread('face.jpg')
    map = sal.compute_saliency('face.jpg')
    # map contains saliency map in double format.

    image = image * map
    image = image.astype(np.uint8)

    cv.imwrite('face_merged,png', image)


