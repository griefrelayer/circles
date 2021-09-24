from os import listdir
import cv2
import numpy as np
from PIL import Image
from raw_pillow_opener import register_raw_opener

import pprint
from sys import argv

register_raw_opener()

pp = pprint.PrettyPrinter(indent=4)

images = listdir('input')

if images[0].endswith('dng'):
    img = np.array(Image.open('input/' + images[0]).convert('RGB'))
    img = img[:, :, ::-1].copy()
else:
    img = cv2.imread('input/' + images[0])


green_channel = img[:, :, 1]
red_channel = img[:, :, 2]
blue_channel = img[:, :, 0]

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if 'bilateral' in argv:
    print('using bilateral filter')
    blurred_green = cv2.bilateralFilter(green_channel, 35, 100, 100)
    blurred_red = cv2.bilateralFilter(red_channel, 35, 100, 100)
    blurred_blue = cv2.bilateralFilter(blue_channel, 35, 100, 100)
else:
    print('using median filter')
    blurred_green = cv2.medianBlur(green_channel, 35)
    blurred_red = cv2.medianBlur(red_channel, 35)
    blurred_blue = cv2.medianBlur(blue_channel, 35)

cv2.imshow('Green after filtering', cv2.resize(blurred_green, (1920, 1080)))
cv2.waitKey(0)

minDist = 120
param1 = 30 #500
param2 = 30 #200 #smaller value-> more false circles
minRadius = 20
maxRadius = 75 #10


circles_prep = cv2.HoughCircles(blurred_green, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                minRadius=minRadius, maxRadius=maxRadius)
circles_prep = np.int16(np.around(circles_prep))
avg = 0
cnt = 0
for i in circles_prep[0, :]:
    avg += i[2]
    cnt += 1
minRadius = int(avg*0.8/cnt)
maxRadius = int(avg*1.2/cnt)

# Docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[,
# maxRadius]]]]]) -> circles
circles_green = cv2.HoughCircles(blurred_green, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                 minRadius=minRadius, maxRadius=maxRadius)

if circles_green is not None:
    circles_green = np.int16(np.around(circles_green))
    print('Green circles num:', circles_green.shape[1])

circles_prep = cv2.HoughCircles(blurred_red, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                minRadius=20, maxRadius=75)
circles_prep = np.int16(np.around(circles_prep))
avg = 0
cnt = 0
for i in circles_prep[0, :]:
    avg += i[2]
    cnt += 1
minRadius = round(avg*0.8/cnt)
maxRadius = round(avg*1.2/cnt)
circles_red = cv2.HoughCircles(blurred_red, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)

if circles_red is not None:
    circles_red = np.int16(np.around(circles_red))
    print('Red circles num:', circles_red.shape[1])

circles_prep = cv2.HoughCircles(blurred_blue, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                minRadius=20, maxRadius=75)
circles_prep = np.int16(np.around(circles_prep))
avg = 0
cnt = 0
for i in circles_prep[0, :]:
    avg += i[2]
    cnt += 1
minRadius = int(avg*0.8/cnt)
maxRadius = int(avg*1.2/cnt)
circles_blue = cv2.HoughCircles(blurred_blue, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                minRadius=minRadius, maxRadius=maxRadius)

if circles_blue is not None:
    circles_blue = np.int16(np.around(circles_blue))
    print('Blue circles num:', circles_blue.shape[1])

green_circles = np.sort(circles_green, axis=1)
red_circles = np.sort(circles_red, axis=1)
blue_circles = np.sort(circles_blue, axis=1)

green_red_common = []
rgb_out = []
red_index = 0
blue_index = 0

for i in range(green_circles.shape[1]):
    c1 = green_circles[:, i][0]
    c2 = red_circles[:, red_index][0]
    while np.any((c1[:2] - c2[:2]) > c1[2]/2):
        red_index += 1
        if red_index >= red_circles.shape[1]:
            break
        c2 = red_circles[:, red_index][0]
    if red_index >= red_circles.shape[1]:
        break
    diff = np.abs(c1[:2] - c2[:2])
    if np.all(diff < c1[2]/2):
        green_red_common.append([[i, red_index], c1[:2] - c2[:2]])

for gr_indices, gr_diff in green_red_common:
    g = gr_indices[0]
    r = gr_indices[1]
    c1 = green_circles[:, g][0]
    c2 = blue_circles[:, blue_index][0]
    while np.any((c1[:2] - c2[:2]) > c1[2]/2):
        blue_index += 1
        if blue_index >= blue_circles.shape[1]:
            break
        c2 = blue_circles[:, blue_index][0]
    if blue_index >= blue_circles.shape[1]:
        break
    diff = np.abs(c1[:2] - c2[:2])
    if np.all(diff < c1[2]/2):
        rgb_out.append([c1[:2].tolist(), gr_diff.tolist(), (c1[:2] - c2[:2]).tolist()])
    # format: [[x, y of center of the green channel circle], [x, y difference between green and red channels],
    # [x, y difference between green and blue channels]

pp.pprint(rgb_out)

for i in circles_red[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)

for i in circles_green[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

for i in circles_blue[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)

cv2.imshow('All circles', cv2.resize(img, (1920, 1080)))
cv2.waitKey(0)

