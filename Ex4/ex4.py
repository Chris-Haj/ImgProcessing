import cv2
import numpy as np

# Load image in grayscale
img_path = 'sudoko.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded properly
if img is None:
    raise ValueError("Image not loaded properly. Please check the path and try again.")

# Convert to float32, a requirement for the Harris corner function
gray = np.float32(img)

# Apply Harris Corner Detection
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Result is dilated for marking the corners, not important for our final image
dst_dilated = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = 0

# Mark corners on the image for visualization
img_corners_marked = img.copy()
img_corners_marked[dst > 0.01 * dst.max()] = 255

# Save and display the resulting image
output_path_corners_marked = 'sudoku_corners_marked.png'
cv2.imwrite(output_path_corners_marked, img_corners_marked)

output_path_corners_marked
