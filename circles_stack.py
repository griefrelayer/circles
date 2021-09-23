from os import listdir
import cv2
import numpy as np


images = listdir('input')

img = cv2.imread('input/' + images[0])
img_green = img.copy()
img_red = img.copy()
img_blue = img.copy()

green_channel = img[:, :, 1]
red_channel = img[:, :, 2]
blue_channel = img[:, :, 0]
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred_green = cv2.medianBlur(cv2.equalizeHist(green_channel), 25) #cv2.bilateralFilter(gray,10,50,50)
blurred_red = cv2.medianBlur(cv2.equalizeHist(red_channel), 25)

minDist = 120
param1 = 30 #500
param2 = 30 #200 #smaller value-> more false circles
minRadius = 20
maxRadius = 75 #10

# docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
circles_green = cv2.HoughCircles(blurred_green, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

if circles_green is not None:
    circles_green = np.uint16(np.around(circles_green))
    for i in circles_green[0, :]:
        cv2.circle(img_green, (i[0], i[1]), i[2], (0, 255, 0), 2)

resized = cv2.resize(img_green, (1920, 1080))
cv2.imshow('img', resized)
cv2.waitKey(0)

circles_red = cv2.HoughCircles(blurred_red, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

if circles_red is not None:
    circles_red = np.uint16(np.around(circles_red))
    for i in circles_red[0, :]:
        cv2.circle(img_red, (i[0], i[1]), i[2], (0, 255, 0), 2)

# Show result for testing:
resized = cv2.resize(img_red, (1920, 1080))
cv2.imshow('img', resized)
cv2.waitKey(0)

cv2.destroyAllWindows()
