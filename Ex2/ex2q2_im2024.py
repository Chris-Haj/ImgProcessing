import cv2 as cv
import numpy as np


def drawRhombus():
    squareSize = 600
    image = np.zeros((squareSize, squareSize), dtype=np.uint8)
    middle = squareSize // 2
    triSize = 150
    for x in range(middle - triSize, middle + triSize):
        for y in range(middle - triSize, middle + triSize):
            if abs(x - middle) + abs(y - middle) < triSize:
                image[x, y] = 255

    return image


def addNoise(image):
    # Copy the image to avoid modifying the original
    noisy_image = np.copy(image)
    rows, cols = noisy_image.shape
    num_noisy_pixels = int(rows * cols * 0.05)  # 5% of the total number of pixels

    # Randomly choose 5% of the pixels
    rand_x = np.random.randint(0, rows, num_noisy_pixels)
    rand_y = np.random.randint(0, cols, num_noisy_pixels)

    # Add Gaussian noise to these pixels
    # Choosing a small std deviation to not overwhelm the image
    noise = np.random.normal(0, 25, num_noisy_pixels)  # Adjust the 25 as needed

    for i in range(num_noisy_pixels):
        noisy_val = noisy_image[rand_x[i], rand_y[i]] + noise[i]
        noisy_image[rand_x[i], rand_y[i]] = np.clip(noisy_val, 0, 255)  # Clip to valid range

    return noisy_image


def edgeDetection(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding width and height
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the original image to handle edges
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Prepare the output image with the same size as the original image
    result = np.zeros(image.shape)

    # Perform the convolution operation
    for i in range(image_height):
        for j in range(image_width):
            # Extract the current region of interest
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and sum the result
            result[i, j] = np.sum(region * kernel)

    return result


if __name__ == '__main__':
    image = drawRhombus()
    noise = addNoise(image)
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    edge = edgeDetection(noise,kernel)
    cv.imshow('beforeNoise', image)
    cv.imshow('afterNoise', noise)
    cv.imshow('edge', edge)

    cv.waitKey(0)
    cv.destroyAllWindows()
