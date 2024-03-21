import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def addNoise(image):

    noisy_image = np.copy(image)
    rows, cols = noisy_image.shape
    num_noisy_pixels = int(rows * cols * 0.05)


    rand_x = np.random.randint(0, rows, num_noisy_pixels)
    rand_y = np.random.randint(0, cols, num_noisy_pixels)

    noise = np.random.normal(0, 25, num_noisy_pixels)

    for i in range(num_noisy_pixels):
        noisy_val = noisy_image[rand_x[i], rand_y[i]] + noise[i]
        noisy_image[rand_x[i], rand_y[i]] = np.clip(noisy_val, 0, 255)

    return noisy_image


def drawRectangle():

    height, width = 300 , 100
    sLength = 50

    image = np.ones((height, width), dtype=np.uint8) * 255

    image[:10, :] = 0
    image[-10:, :] = 0
    image[:, :10] = 0
    image[:, -10:] = 0

    topEdge = height // 5
    bottomEdge = height - topEdge
    halfLength = sLength // 2
    middleWid = width // 2
    image[topEdge:topEdge + sLength, middleWid - halfLength:middleWid + halfLength] = 0
    image[bottomEdge - sLength:bottomEdge, middleWid - halfLength:middleWid + halfLength] = 0

    return image


def apply_gaussian_blur(image):
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0

    output_image = np.zeros_like(image)

    k_height, k_width = kernel.shape
    pad_height, pad_width = k_height // 2, k_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)


    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            output_image[row, col] = np.sum(kernel * padded_image[row:row + k_height, col:col + k_width])

    return output_image

def findEdges(image):

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


    edges_x = np.zeros_like(image, dtype=np.float32)
    edges_y = np.zeros_like(image, dtype=np.float32)


    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)


    for row in range(1, image.shape[0] + 1):
        for col in range(1, image.shape[1] + 1):

            edges_x[row - 1, col - 1] = np.sum(sobel_x * padded_image[row - 1:row + 2, col - 1:col + 2])

            edges_y[row - 1, col - 1] = np.sum(sobel_y * padded_image[row - 1:row + 2, col - 1:col + 2])


    sobelImg = np.hypot(edges_x, edges_y)
    sobelImg = sobelImg / sobelImg.max() * 255
    sobelImg = np.clip(sobelImg, 0, 255).astype(np.uint8)

    return sobelImg




if __name__ == '__main__':
    image = drawRectangle()
    noise = addNoise(image)
    removeNoise = apply_gaussian_blur(noise)
    detectedEdges = findEdges(removeNoise)
    cv.imshow('original', image)
    cv.imshow('noise', noise)
    cv.imshow('gausian', removeNoise)
    cv.imshow('foundEdges', detectedEdges)
    cv.waitKey(0)
    cv.destroyAllWindows()
