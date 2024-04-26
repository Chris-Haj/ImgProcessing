import numpy as np
import cv2
from numpy.fft import ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

def apply_fft_decompression(compressed_image_path):
    # Assuming the compressed data is stored in a way that it can be loaded directly
    # For this example, let's pretend we're working directly with the FFT data
    # In practice, you'd need to load this from storage
    f_shift = np.load(compressed_image_path)  # This is a placeholder step

    # Apply inverse FFT without applying a mask
    f_ishift = ifftshift(f_shift)
    img_reconstructed = ifft2(f_ishift)
    img_reconstructed = np.abs(img_reconstructed)

    # Display the reconstructed image
    plt.figure(figsize=(6, 6))
    plt.imshow(img_reconstructed, cmap='gray'), plt.title('Reconstructed Image')
    plt.show()

    # Optionally, save the reconstructed image
    reconstructed_image_path = "reconstructed_image.jpg"
    cv2.imwrite(reconstructed_image_path, img_reconstructed)
    print(f"Reconstructed image saved as {reconstructed_image_path}")

# Example usage, assuming the compressed data is stored in an appropriate format
apply_fft_decompression('compressed_image.jpg')
