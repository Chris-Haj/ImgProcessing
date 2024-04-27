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


def loadImageIntoGrayScale(imagePath, fx=0.5, fy=0.5):
    image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError('Image not found')
    image = cv.resize(image, None, None, fx, fy, cv.INTER_LINEAR)
    return image

if __name__ == '__main__':
    ImagesPaths = [image for image in Images.values()]
    GrayImages = [loadImageIntoGrayScale(image) for image in ImagesPaths[:2]]

    for image in GrayImages:
        clahe = cv.createCLAHE(5, (5, 10))
        newImage = clahe.apply(src=image)
        plt.imshow(image, cmap='gray')
        plt.show()
        plt.imshow(newImage, cmap='gray')
        plt.show()

