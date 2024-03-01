import cv2 as cv
import numpy as np


def drawRhombus():
    squareSize = 200
    image = np.zeros((squareSize, squareSize), dtype=np.uint8)
    middle = squareSize // 2
    triSize = 50
    for x in range(middle - triSize, middle + triSize):
        for y in range(middle - triSize, middle + triSize):
            if abs(x - middle) + abs(y - middle) < triSize:
                image[x, y] = 255

    return image


def edgeDetection(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    result = np.zeros(image.shape)

    for i in range(image_height):
        for j in range(image_width):

            region = padded_image[i:i + kernel_height, j:j + kernel_width]

            result[i, j] = np.sum(region * kernel)

    return result


if __name__ == '__main__':
    image = drawRhombus()
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    res = edgeDetection(image, kernel)
    cv.imshow('rhombus', image)
    cv.imshow('result', res)
    cv.waitKey(0)
    cv.destroyAllWindows()
