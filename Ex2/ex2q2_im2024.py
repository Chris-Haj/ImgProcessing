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
    # Copy the image to avoid modifying the original
    noisy_image = np.copy(image)
    rows, cols = noisy_image.shape
    num_noisy_pixels = int(rows * cols * 0.05)  # 5% of the total number of pixels

    # Randomly choose 5% of the pixels
    rand_x = np.random.randint(0, rows, num_noisy_pixels)
    rand_y = np.random.randint(0, cols, num_noisy_pixels)

    # Add Gaussian noise to these pixels
    # Choosing a small std deviation to not overwhelm the image
    noise = np.random.normal(0, 15, num_noisy_pixels)  # Adjust the 25 as needed

    for i in range(num_noisy_pixels):
        noisy_val = noisy_image[rand_x[i], rand_y[i]] + noise[i]
        noisy_image[rand_x[i], rand_y[i]] = np.clip(noisy_val, 0, 255)  # Clip to valid range

    return noisy_image


def convolve(image, kernel):
    """
    Manually apply a 2D convolution operation, without using any libraries that perform the operation.
    """
    # Kernel needs to be flipped both horizontally and vertically
    kernel = np.flipud(np.fliplr(kernel))

    # Gather the dimensions to define the size of the output image
    xImage = image.shape[0]
    yImage = image.shape[1]
    xKernel = kernel.shape[0]
    yKernel = kernel.shape[1]

    # Define the output image matrix, ensuring to adjust for the loss of border pixels
    output = np.zeros((xImage - xKernel + 1, yImage - yKernel + 1))

    # Execute convolution operation (manual implementation)
    for x in range(xImage - xKernel + 1):
        for y in range(yImage - yKernel + 1):
            # Element-wise multiplication and summation
            output[x, y] = np.sum(image[x:x + xKernel, y:y + yKernel] * kernel)

    return output

def apply_sobel_operator(image):
    """
    Apply the Sobel operator to an image to detect edges.
    """
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
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return magnitude



def apply_gaussian_blur(image):
    # Assuming kernel is a 2D numpy array for Gaussian blur
    # Initialize the output image
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0

    output_image = np.zeros_like(image)

    # Kernel dimensions
    k_height, k_width = kernel.shape
    pad_height, pad_width = k_height // 2, k_width // 2

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Convolve the kernel with the image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            output_image[row, col] = np.sum(kernel * padded_image[row:row + k_height, col:col + k_width])

    return output_image


if __name__ == '__main__':
    image = drawRhombus()
    noise = addNoise(image)
    edge = apply_gaussian_blur(noise)
    cv.imwrite('noisy.png', noise)
    cv.imwrite('findEdges.png', apply_sobel_operator(edge))

