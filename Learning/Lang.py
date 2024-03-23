import cv2 as cv
import pytesseract as tess


def getText(lang):
    img = cv.imread('img_1.png')
    text = tess.image_to_string(image=img, lang=lang)
    return text


if __name__ == '__main__':
    text = getText('heb')
    print(text)
