import cv2 as cv
import numpy as np


def applyOtsuThresh(Image):
    # hist_eq_image = cv.equalizeHist(Image)
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

    heightA, widthA = imgA.shape
    heightB, widthB = imgB.shape
    heightC, widthC = imgC.shape

    avg_height = int((heightA + heightB + heightC) / 3)
    avg_width = int((widthA + widthB + widthC) / 3)

    imgA = cv.resize(imgA, (avg_width, avg_height))
    imgB = cv.resize(imgB, (avg_width, avg_height))
    imgC = cv.resize(imgC, (avg_width, avg_height))

    otsuA, otsuB, otsuC = applyOtsuThresh(imgA), applyOtsuThresh(imgB), applyOtsuThresh(imgC)
    AllComparisons = np.vstack((otsuA, otsuB, otsuC))
    AllComparisons = cv.resize(AllComparisons, None, None, 0.5, 0.5, cv.INTER_LINEAR)
    cv.imwrite('A_B_C Before and After.jpg', AllComparisons)
    cv.imshow('A_B_C Before and After', AllComparisons)

cv.waitKey(0)
cv.destroyAllWindows()
