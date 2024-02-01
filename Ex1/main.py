import cv2 as cv
import numpy as np


def applyOtsuThresh(Image):
    bilnearImage = Image
    for i in range(10):
        bilnearImage = cv.resize(bilnearImage, None, None, 0.5, 0.5, cv.INTER_LINEAR)
        bilnearImage = cv.resize(bilnearImage, None, None, 2, 2, cv.INTER_LINEAR)
    #  clahe = cv.createCLAHE(3, (5, 10))
    clahe = cv.createCLAHE(15, (5, 10))
    newImage = clahe.apply(src=bilnearImage)
    bilnearImage = newImage
    for i in range(5):
        bilnearImage = cv.resize(bilnearImage, None, None, 0.5, 0.5, cv.INTER_LINEAR)
        bilnearImage = cv.resize(bilnearImage, None, None, 2, 2, cv.INTER_LINEAR)

    thresh, newImage = cv.threshold(bilnearImage, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    ComparedImages = np.hstack((Image, newImage))

    return ComparedImages


if __name__ == '__main__':
    imgA = cv.imread('A.jpg', cv.IMREAD_GRAYSCALE)
    imgB = cv.imread('B.jpg', cv.IMREAD_GRAYSCALE)
    imgC = cv.imread('C.jpg', cv.IMREAD_GRAYSCALE)

    otsuA, otsuB, otsuC = applyOtsuThresh(imgA), applyOtsuThresh(imgB), applyOtsuThresh(imgC)
    cv.imwrite('newA.jpg', otsuA)
    cv.imwrite('newB.jpg', otsuB)
    cv.imwrite('newC.jpg', otsuC)


