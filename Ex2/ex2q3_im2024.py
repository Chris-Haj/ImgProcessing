import cv2 as cv
import numpy as np

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

def drawRectangle():
    # Image dimensions
    height, width = 300*2, 100*2
    sLength = 50*2  # Side length for both squares

    # Create a white image
    image = np.ones((height, width), dtype=np.uint8) * 255

    # Draw the outer rectangle border, 10 pixels thick
    image[:10, :] = 0  # Top border
    image[-10:, :] = 0  # Bottom border
    image[:, :10] = 0  # Left border
    image[:, -10:] = 0  # Right border

    topEdge = height // 5
    bottomEdge = height - topEdge
    halfLength = sLength // 2
    middleWid = width // 2
    image[topEdge:topEdge+sLength, middleWid - halfLength:middleWid + halfLength] = 0
    image[bottomEdge-sLength:bottomEdge, middleWid - halfLength:middleWid + halfLength] = 0


    return image


if __name__ == '__main__':
    image = drawRectangle()
    noise = addNoise(image)
    cv.imwrite('image.png', image)
    cv.imwrite('noise.png', noise)
