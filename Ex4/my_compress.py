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

    # Zeroing out the high frequencies
    # Here, instead of a mask, we directly modify the FFT shifted matrix
    f_shift[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0

    # Creating an outline for the preserved frequencies
    mask = np.zeros(img.shape, np.uint8)
    cv2.rectangle(mask, (ccol - cutoff_frequency, crow - cutoff_frequency), (ccol + cutoff_frequency, crow + cutoff_frequency), 255, 2)
    blue_square = cv2.merge([img, img, img])  # Converting grayscale to BGR
    blue_square[mask == 255] = [0, 0, 255]  # Applying blue color to the outline

    # Applying inverse FFT
    f_ishift = ifftshift(f_shift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Show original and compressed image
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(img_back, cmap='gray'), plt.title('Compressed Image')
    plt.figure(), plt.imshow(blue_square), plt.title('Preserved Frequencies with Blue Outline')
    plt.show()

    # Save the compressed image
    compressed_image_path = "compressed_image.jpg"
    cv2.imwrite(compressed_image_path, img_back)
    print(f"Compressed image saved as {compressed_image_path}")


# Example usage
apply_fft_compression('img.png', 30)
