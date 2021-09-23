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

blurred_green = cv2.medianBlur(green_channel, 35) #cv2.bilateralFilter(gray,10,50,50)
blurred_red = cv2.medianBlur(red_channel, 35)
blurred_blue = cv2.medianBlur(blue_channel, 35)
'''
# apply basic thresholding -- the first parameter is the image
# we want to threshold, the second value is is our threshold
# check; if a pixel value is greater than our threshold (in this
# case, 200), we set it to be *black, otherwise it is *white*
(T, threshInv) = cv2.threshold(blurred_green, 100, 200, cv2.THRESH_BINARY_INV)
# cv2.imshow("Threshold Binary Inverse", threshInv)
cv2.imshow('blurred_green', cv2.resize(threshInv, (1920, 1080)))
cv2.waitKey(0)
'''
minDist = 120
param1 = 30 #500
param2 = 30 #200 #smaller value-> more false circles
minRadius = 20
maxRadius = 75 #10


circles_prep = cv2.HoughCircles(blurred_green, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                minRadius=minRadius, maxRadius=maxRadius)
circles_prep = np.uint16(np.around(circles_prep))
avg = 0
cnt = 0
for i in circles_prep[0, :]:
    avg += i[2]
    cnt += 1
minRadius = int(avg*0.8/cnt)
maxRadius = int(avg*1.2/cnt)
# docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
circles_green = cv2.HoughCircles(blurred_green, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

if circles_green is not None:
    circles_green = np.uint16(np.around(circles_green))
    print(circles_green.shape)

circles_prep = cv2.HoughCircles(blurred_red, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                minRadius=20, maxRadius=75)
circles_prep = np.uint16(np.around(circles_prep))
avg = 0
cnt = 0
for i in circles_prep[0, :]:
    avg += i[2]
    cnt += 1
minRadius = round(avg*0.8/cnt)
maxRadius = round(avg*1.2/cnt)
circles_red = cv2.HoughCircles(blurred_red, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

if circles_red is not None:
    circles_red = np.uint16(np.around(circles_red))
    print(circles_red.shape)

circles_prep = cv2.HoughCircles(blurred_blue, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                minRadius=20, maxRadius=75)
circles_prep = np.uint16(np.around(circles_prep))
avg = 0
cnt = 0
for i in circles_prep[0, :]:
    avg += i[2]
    cnt += 1
minRadius = int(avg*0.8/cnt)
maxRadius = int(avg*1.2/cnt)
circles_blue = cv2.HoughCircles(blurred_blue, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

if circles_blue is not None:
    circles_blue = np.uint16(np.around(circles_red))
    print(circles_blue.shape)

green_circles = np.sort(circles_green, axis=1)
red_circles = np.sort(circles_red, axis=1)
blue_circles = np.sort(circles_blue, axis=1)


