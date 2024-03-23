import numpy as np
import cv2
from numpy.fft import ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


def decode_dims_from_img(img):
    """Extract the original dimensions from the embedded data."""
    # Assuming the dimensions are correctly extracted as before
    dims_bin = ''
    for i in range(32):
        row = i // img.shape[1]
        col = i % img.shape[1]
        pixel_bin = '{:08b}'.format(img[row, col])
        dims_bin += pixel_bin[-1]
    height = int(dims_bin[:16], 2)
    width = int(dims_bin[16:], 2)
    return height, width


def apply_fft_decompression(image_path):
    # Load the compressed image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Decode the original dimensions (assuming this part works as intended)
    original_height, original_width = decode_dims_from_img(img)

    # Assuming the rest of the image beyond the first 32 pixels is untouched FFT data ready for IFFT
    # Correct the preparation for IFFT
    f_ishift = ifftshift(fftshift(img))
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize the image data to 0-255 range and convert to uint8
    img_back = (img_back / np.max(img_back) * 255).astype(np.uint8)

    # Crop to the original dimensions
    img_decompressed = img_back[:original_height, :original_width]

    # Show decompressed image
    plt.imshow(img_decompressed, cmap='gray')
    plt.title('Decompressed Image')
    plt.show()


# Example usage
apply_fft_decompression('compressed_image_with_dims.jpg')
