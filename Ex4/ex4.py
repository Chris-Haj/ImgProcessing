import cv2
import numpy as np


def detect_sudoku_edges(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, 0)

    # Convert to float32, a requirement for the Harris corner function
    gray = np.float32(img)

    # Apply Harris Corner Detection
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Result is dilated for marking the corners
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, marking the corners in the original image
    img[dst > 0.01 * dst.max()] = [255,0,255]  # This will mark corners with black dots

    # Display the image with corners marked
    cv2.imshow('Harris Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
detect_sudoku_edges('sudoko.png')
