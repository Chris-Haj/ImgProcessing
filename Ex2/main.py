import cv2 as cv
import numpy as np

if __name__ == '__main__':
    squareSize = 200
    image = np.zeros((squareSize, squareSize), dtype=np.uint8)
    middle = squareSize // 2
    triSize = 50
    for i in range(triSize):
        image[middle,middle-triSize:middle+triSize] = 255

    cv.imwrite('test.png', image)
