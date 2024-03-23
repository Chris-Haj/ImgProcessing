import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv.imread('rice-shaded.tif', cv.IMREAD_GRAYSCALE)
    img = to_ndarray(...)
    img1 = gs_filter(img...)
    img2, D = gradient_intensity(img1)
    img3 = suppression(np.copy(img2), D)
    img4, weak = threshold(np.copy(img3), ..., ...)
    img5 = tracking(np.copy(img4), weak)
