import cv2
import numpy as np


# Define a function to apply Otsu's thresholding
def apply_otsus_thresholding(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded properly
    if image is None:
        return None, "Could not read the image."

    # Apply Gaussian blur to reduce noise and improve thresholding
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Otsu's thresholding after Gaussian filtering
    _, otsu_thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu_thresh, None


# Reapply the function to the three images uploaded earlier (A, B, and C) and save the results.
# We also need to re-define the image paths
image_paths = {
    'A': 'A.jpg',
    'B': 'B.jpg',
    'C': 'C.jpg'
}

# Dictionary to store output paths or errors
processed_images = {}

# Process each image and store the results
for label, path in image_paths.items():
    processed_image, error = apply_otsus_thresholding(path)
    if processed_image is not None:
        # Save the processed image
        output_path = f'processed_{label}.jpg'
        cv2.imwrite(output_path, processed_image)
        processed_images[f'processed_{label}'] = output_path
    else:
        processed_images[f'error_{label}'] = error

print(processed_images)