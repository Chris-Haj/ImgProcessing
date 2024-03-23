import numpy as np
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


def int_to_bin(rgb):
    """Convert an integer tuple to a binary (string) tuple."""
    r, g, b = rgb
    return ('{0:08b}'.format(r),
            '{0:08b}'.format(g),
            '{0:08b}'.format(b))


def bin_to_int(rgb):
    """Convert a binary (string) tuple to an integer tuple."""
    r, g, b = rgb
    return (int(r, 2),
            int(g, 2),
            int(b, 2))


def encode_dims_in_img(img, dims):
    """Encode the image dimensions in the first 16 pixels of the image (assuming dimensions can fit within this)."""
    # Convert dimensions to binary
    dims_bin = '{:016b}'.format(dims[0]) + '{:016b}'.format(dims[1])  # 16 bits for each dimension
    for i in range(32):
        row = i // img.shape[1]
        col = i % img.shape[1]
        pixel_bin = '{:08b}'.format(img[row, col])
        # Replace the least significant bit with a bit from the dimensions
        new_pixel_bin = pixel_bin[:-1] + dims_bin[i]
        img[row, col] = int(new_pixel_bin, 2)
    return img


def apply_fft_compression(image_path, cutoff_frequency):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_dims = img.shape  # Save original dimensions

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

    # Plot original and compressed image before embedding dimensions
    plt.figure(figsize=(18, 6))
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(132), plt.imshow(img_back, cmap='gray'), plt.title('Compressed Image')

    # Encode dimensions in the image
    img_back_encoded = encode_dims_in_img(img_back.astype(np.uint8), original_dims)

    # Plot the compressed image with dimensions encoded
    plt.subplot(133), plt.imshow(img_back_encoded, cmap='gray'), plt.title('Compressed with Dims Encoded')
    plt.show()

    # Save the compressed image with dimensions encoded
    compressed_image_path = "compressed_image_with_dims.jpg"
    cv2.imwrite(compressed_image_path, img_back_encoded)
    print(f"Compressed image saved as {compressed_image_path}")


# Example usage
apply_fft_compression('img.png', 30)
