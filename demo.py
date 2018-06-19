from Salicon import Salicon
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    sal = Salicon()
    image = cv.imread('face.jpg')
    height, width = image.shape[:2]
    map = sal.compute_saliency('face.jpg')
    # map contains saliency map in double format.

    print('np.max(map): ' + str(np.max(map)))
    print('np.min(map): ' + str(np.min(map)))

    map = np.reshape(map, (height, width, 1))
    image = image * map
    image = image.astype(np.uint8)

    cv.imwrite('face_merged,png', image)


