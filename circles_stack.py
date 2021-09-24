from os import listdir
import cv2
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)

images = listdir('input')

img = cv2.imread('input/' + images[0])
'''img_green = img.copy()
img_red = img.copy()
img_blue = img.copy()'''

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
circles_prep = np.int16(np.around(circles_prep))
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
circles_red = cv2.HoughCircles(blurred_red, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

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
circles_blue = cv2.HoughCircles(blurred_blue, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

if circles_blue is not None:
    circles_blue = np.int16(np.around(circles_red))
    print('Blue circles num:', circles_blue.shape[1])

green_circles = np.sort(circles_green, axis=1)
red_circles = np.sort(circles_red, axis=1)
blue_circles = np.sort(circles_blue, axis=1)
print(green_circles)

green_red_common = []
rgb_out = []
red_index = 0
blue_index = 0
d1 = green_circles[:, 1][0]
print(d1)
d2 = red_circles[:, 1][0]
print(d2)
d3 = blue_circles[:, 1][0]
print(d3)

print('np.sum(c1[:2] - c2[:2]) = ', np.sum(d1[:2] - d2[:2]))
print('c1 radius:', d1[2])

for i in range(green_circles.shape[1]):
    c1 = green_circles[:, i][0]
    c2 = red_circles[:, red_index][0]
    if red_index < 15:
        print("c1:", c1)
    while np.any((c1[:2] - c2[:2]) > c1[2]/2):
        red_index += 1

        if red_index >= red_circles.shape[1]:
            break
        c2 = red_circles[:, red_index][0]
        if red_index < 15:
            print("c2:", c2)
            print('c1 - c2 = ', c1[:2] - c2[:2])
    if red_index >= red_circles.shape[1]:
        break
    diff = np.abs(c1[:2] - c2[:2])
    if np.all(diff < c1[2]/2):
        green_red_common.append([[i, red_index], c1[:2] - c2[:2]])

# print("green + red circles difference", green_red_common)

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
    # format: [[x, y of center of the green channel circle], [x, y difference between green and red channels], [x, y difference between green and blue channels]

pp.pprint(rgb_out)

