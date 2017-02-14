import numpy as np
import cv2
from matplotlib import pyplot as plt

def conversion():
    image = cv2.imread('pestimg.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.png',gray_image)
    cv2.imshow('color_image',image)
    cv2.imshow('gray_image',gray_image) 
    cv2.waitKey(0)                 # Waits forever for user to press any key   
    cv2.destroyAllWindows()
def blur():
    image = cv2.imread('gray_image.png')
 
    cv2.getGaussianKernel(5,5)
    blur=cv2.GaussianBlur(image,(5,5),0)
    cv2.imwrite('blur.png',blur)
    cv2.imshow('blur.png',blur)
    cv2.waitKey(0)                 # Waits forever for user to press any key
    cv2.destroyAllWindows()
def average():
    img = cv2.imread('blur.png')
     
    blur = cv2.blur(img,(5,5))
     
    plt.subplot(121),plt.imshow(blur),plt.title('blur')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()
def segmentation():
    
    img = cv2.imread('blurred.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

conversion()    
blur()
average()
segmentation()
