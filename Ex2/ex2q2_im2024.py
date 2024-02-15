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


if __name__ == '__main__':
    image = drawRhombus()
    noise = addNoise(image)
    cv.imshow('beforeNoise',image)
    cv.imshow('afterNoise',noise)
    cv.waitKey(0)
    cv.destroyAllWindows()
