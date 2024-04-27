import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv.imread('rice-shaded.tif', cv.IMREAD_GRAYSCALE)
    img = cv.to_ndarray(...)
    img1 = cv.gs_filter(img)
    img2, D = cv.gradient_intensity(img1)
    img3 = cv.suppression(np.copy(img2), D)
    img4, weak = cv.threshold(np.copy(img3), ..., ...)
    img5 = cv.tracking(np.copy(img4), weak)
