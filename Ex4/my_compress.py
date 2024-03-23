import numpy as np
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

def apply_fft_compression(image_path, cutoff_frequency):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply FFT
    f_transform = fft2(img)
    f_shift = fftshift(f_transform)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask with the same dimensions as the image, with all zeros
    mask = np.zeros((rows, cols), np.uint8)
    # Create a circular mask to preserve low frequencies within the cutoff_frequency radius
    cv2.circle(mask, (ccol, crow), cutoff_frequency, 1, thickness=-1)
    # Apply the mask by using element-wise multiplication
    f_shift = f_shift * mask

    # Applying inverse FFT
    f_ishift = ifftshift(f_shift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Show original and compressed image
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(img_back, cmap='gray'), plt.title('Compressed Image')
    plt.show()

    # Save the compressed image
    compressed_image_path = "compressed_image.jpg"
    cv2.imwrite(compressed_image_path, img_back)
    print(f"Compressed image saved as {compressed_image_path}")

# Example usage
apply_fft_compression('img.png', 30)
