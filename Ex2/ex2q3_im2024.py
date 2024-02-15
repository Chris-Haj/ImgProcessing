import cv2 as cv
import numpy as np


def drawRectangle():
    # Image dimensions
    height, width = 300, 100
    sLength = 50  # Side length for both squares

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
    cv.imwrite('image.png', image)
