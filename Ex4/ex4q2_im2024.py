import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Load image in grayscale
img_path = 'sudoko.png'
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

# Check if the image is loaded properly
if img is None:
    raise ValueError("Image not loaded properly. Please check the path and try again.")

# Convert to float32 for Gaussian Blur (keeping original for drawing)
grayImg = np.float32(img)
gaus = cv.GaussianBlur(grayImg, (9, 9), 0)

# Convert back to uint8 for Canny edge detector
gaus_uint8 = np.uint8(gaus)
canny = cv.Canny(gaus_uint8, 50, 150)

# Prepare a black canvas to draw lines
blackBox = np.zeros(img.shape, dtype=np.uint8)

# Detect lines
lines = cv.HoughLines(canny, 1, np.pi / 180, threshold=200)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(blackBox, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Detect corners on the original grayscale image
corners = cv.cornerHarris(np.float32(img), blockSize=2, ksize=3, k=0.04)

# Result is dilated for marking the corners

# Result is dilated for marking the corners
dilated_corners = cv.dilate(corners, None)
finalRes = np.stack([grayImg, grayImg,grayImg], axis=-1)
finalRes[corners > 0.1 * corners.max()] = [255, 0, 0]

# Display
plt.imshow(finalRes, cmap='gray')

# Display
plt.show()
