import cv2 as cv
import skimage as ski
import matplotlib.pyplot as plt

Images = {0.05: './CoinImages/0.05Francs.jpg',
          0.1: './CoinImages/0.1Francs.jpg',
          0.2: './CoinImages/0.2Francs.jpg',
          5: './CoinImages/5Francs.jpg'}

ProcessedImages = ['./ProcessedImages/Processed0.05Francs.jpg',
                   './ProcessedImages/Processed0.1Francs.jpg',
                   './ProcessedImages/Processed0.2Francs.jpg',
                   './ProcessedImages/Processed5Francs.jpg']

"""
1. Preprocessing
Convert to Grayscale: Coins have distinct textures and edges which can be more easily
detected in a grayscale image.
Noise Reduction: Apply a Gaussian blur or median filter to reduce image noise without affecting edge information significantly.
"""
"""
2. Edge Detection
Use edge detection algorithms like the Canny edge detector to find the boundaries of coins.
"""

"""
3. Hough Circle Transform
Apply the Hough Circle Transform to detect circles in the preprocessed image
as coins will typically appear as circles from above.
"""
"""
4. Segmentation and Feature Extraction
Once circles are detected, segment each coin from the image.
Extract features that can help identify the coin, such as radius
of the detected circle, color information (if any), and texture.
"""

"""
5. Classification
Based on the extracted features, classify the coins. 
This may involve a simple rule-based approach (e.g., radius size)
or a more sophisticated classification algorithm.
"""

"""
6. Handling Overlaps
For overlapping coins, edge information might not be enough. 
You might need to look at contours and the interruption in the
continuity of edges to infer overlaps.
Segmenting overlapping coins accurately can be complex and may require advanced techniques
like contour analysis or training a machine learning model.
"""

"""
7. Output
The final step is to output the denomination and count of each coin.
For the images with multiple coins and partial coverings, the segmentation and 
classification can become significantly more complicated, 
as standard Hough Circle Transform may not work well with partial circles.
"""


def loadImageIntoGrayScale(imagePath, fx=0.5, fy=0.5):
    image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError('Image not found')
    image = cv.resize(image, None, None, fx, fy, cv.INTER_LINEAR)
    return image


def openClose(image, kernelSize=3, iterations=1):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernelSize, kernelSize))
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=iterations)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=iterations)
    return image


if __name__ == '__main__':
    ImagesPaths = [image for image in Images.values()][0:1]
    GrayImages = [loadImageIntoGrayScale(image, 1, 1) for image in ImagesPaths]
    OriginalSizes = [image.shape for image in GrayImages]

    for image in GrayImages:
        plt.imshow(image, cmap='gray')
        plt.show()
        clahe = cv.createCLAHE(15, (5, 10))
        image = openClose(image, iterations=10)
        image = clahe.apply(src=image)
        image = openClose(image, iterations=10)
        thresh, newImage = cv.threshold(image, 0, 200, cv.THRESH_BINARY + cv.THRESH_OTSU)
        plt.imshow(newImage, cmap='gray')
        plt.show()
