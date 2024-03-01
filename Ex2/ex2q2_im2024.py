import cv2 as cv
import numpy as np

def drawRhombus():
    factor = 2
    squareSize = 200 * factor
    image = np.zeros((squareSize, squareSize), dtype=np.uint8)
    middle = squareSize // 2
    triSize = 50 * factor
    for x in range(middle - triSize, middle + triSize):
        for y in range(middle - triSize, middle + triSize):
            if abs(x - middle) + abs(y - middle) < triSize:
                image[x, y] = 255
    return image

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

def convolve(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    xImage = image.shape[0]
    yImage = image.shape[1]
    xKernel = kernel.shape[0]
    yKernel = kernel.shape[1]

    output = np.zeros((xImage - xKernel + 1, yImage - yKernel + 1))

    for x in range(xImage - xKernel + 1):
        for y in range(yImage - yKernel + 1):

            output[x, y] = np.sum(image[x:x + xKernel, y:y + yKernel] * kernel)

    return output

def apply_sobel_operator(image):

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    gradient_x = convolve(image, sobel_x)
    gradient_y = convolve(image, sobel_y)
    sobelImage = gradient_x**2 + gradient_y**2
    return sobelImage

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

if __name__ == '__main__':
    image = drawRhombus()
    noise = addNoise(image)
    edge = apply_gaussian_blur(noise)
    cv.imshow('image', image)
    cv.imshow('noisy', noise)
    cv.imshow('findEdges', apply_sobel_operator(edge))
    cv.waitKey(0)
    cv.destroyAllWindows()

