import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv.imread('rice-shaded.tif', cv.IMREAD_GRAYSCALE)
    disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    erodeImg = cv.erode(img, disk)
    dilateImg = cv.dilate(img, disk)
    gradiantImg = cv.absdiff(dilateImg, erodeImg)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(cv.cvtColor(gradiantImg, cv.COLOR_BGR2RGB))
    ax[1].set_title('Edges')
    ax[1].axis('off')
    plt.show()
