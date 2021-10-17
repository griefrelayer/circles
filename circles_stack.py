from os import listdir
import cv2
import numpy as np
from PIL import Image
from raw_pillow_opener import register_raw_opener

import pprint
from sys import argv

debug = True
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

'''cv2.imshow('Green after filtering', cv2.resize(blurred_green, (1920, 1080)))
cv2.waitKey(0)
cv2.imshow('Red after filtering', cv2.resize(blurred_red, (1920, 1080)))
cv2.waitKey(0)
cv2.imshow('Blue after filtering', cv2.resize(blurred_blue, (1920, 1080)))
cv2.waitKey(0)'''

minDist = 100
param1 = 30  # 500
param2 = 30  # 200 #smaller value-> more false circles
orig_minRadius = 20
orig_maxRadius = 75  # 10


def get_min_max_radius(blurred_channel, minRadius, maxRadius):
    circles_prep = cv2.HoughCircles(blurred_channel, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                    minRadius=minRadius, maxRadius=maxRadius)
    circles_prep = np.int16(np.around(circles_prep))
    avg = 0
    cnt = 0
    for i in circles_prep[0, :]:
        avg += i[2]
        cnt += 1
    minRadius = int(avg * 0.8 / cnt)
    maxRadius = int(avg * 1.2 / cnt)
    return minRadius, maxRadius


def get_circles(blurred_channel, minRadius, maxRadius):
    return cv2.HoughCircles(blurred_channel, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                            minRadius=minRadius, maxRadius=maxRadius)


minRadius, maxRadius = get_min_max_radius(blurred_green, orig_minRadius, orig_maxRadius)

# Docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[,
# maxRadius]]]]]) -> circles
circles_green = get_circles(blurred_green, minRadius, maxRadius)

if circles_green is not None:
    circles_green = np.int16(np.around(circles_green))
    if debug:
        print('Green circles num:', circles_green.shape[1])

minRadius, maxRadius = get_min_max_radius(blurred_red, orig_minRadius, orig_maxRadius)
circles_red = get_circles(blurred_red, minRadius, maxRadius)

if circles_red is not None:
    circles_red = np.int16(np.around(circles_red))
    if debug:
        print('Red circles num:', circles_red.shape[1])

minRadius, maxRadius = get_min_max_radius(blurred_blue, orig_minRadius, orig_maxRadius)
circles_blue = get_circles(blurred_blue, minRadius, maxRadius)

if circles_blue is not None:
    circles_blue = np.int16(np.around(circles_blue))
    if debug:
        print('Blue circles num:', circles_blue.shape[1])


def sorting(c):
    return c[np.lexsort((c[:, 1], c[:, 0]))]


green_circles = sorting(circles_green[0])
red_circles = sorting(circles_red[0])
blue_circles = sorting(circles_blue[0])
if debug:
    print(green_circles[:10])
    print(red_circles[:10])

green_red_common = []
rgb_out = []
red_index = 0
blue_index = 0
i = 0

while 1:
    if red_index >= red_circles.shape[0] or i >= green_circles.shape[0]:
        break
    c1 = green_circles[i]
    c2 = red_circles[red_index]
    if debug:
        print(c1, c2, "<= got these points")
    diff = c1[:2] - c2[:2]
    while diff[1] > c1[2] > abs(diff[0]):
        red_index += 1
        if red_index >= red_circles.shape[0] or i >= green_circles.shape[0]:
            break
        c2 = red_circles[red_index]
        diff = c1[:2] - c2[:2]
        if debug:
            print(c1, c2, "<= inside first while")

    while diff[1] < -c1[2] < - abs(diff[0]):
        i += 1
        if i >= green_circles.shape[0]:
            break
        c1 = green_circles[i]
        diff = c1[:2] - c2[:2]
        if debug:
            print(c1, c2, "<= inside second while")

    if np.all(np.abs(diff) < c1[2]):
        if debug:
            print("adding point...")
        green_red_common.append([[i, red_index], c1[:2] - c2[:2]])
        i += 1
        red_index += 1
    elif diff[1] > c1[2]:
        red_index += 1
    else:
        i += 1

if debug:
    print(len(green_red_common))

print("="*70)
print("="*70)

i = 0
while 1:
    if blue_index >= blue_circles.shape[0] or i >= len(green_red_common):
        break
    gr_indices, gr_diff = green_red_common[i]
    g = gr_indices[0]
    c1 = green_circles[g]
    c2 = blue_circles[blue_index]
    if debug:
        print(c1, c2, "<= got these points")
    diff = c1[:2] - c2[:2]
    while diff[1] > c1[2] > abs(diff[0]):
        if debug:
            print(c1, c2, "<= inside first while (blue)")
        blue_index += 1
        if blue_index >= blue_circles.shape[0]:
            break
        c2 = blue_circles[blue_index]
        diff = c1[:2] - c2[:2]
    while diff[1] < -c1[2] < - abs(diff[0]):
        if debug:
            print(c1, c2, "<= inside second while (blue)")
        i += 1
        if i >= len(green_red_common):
            break
        gr_indices, gr_diff = green_red_common[i]
        g = gr_indices[0]
        c1 = green_circles[g]
        diff = c1[:2] - c2[:2]
    if np.all(np.abs(diff) < c1[2]):
        if debug:
            print("adding point (blue)...")
        rgb_out.append([c1[:2].tolist(), gr_diff.tolist(), (c1[:2] - c2[:2]).tolist()])
        i += 1
        blue_index += 1
    elif diff[1] > c1[2]:
        blue_index += 1
    else:
        i += 1
    # format: [[x, y of center of the green channel circle], [x, y difference between green and red channels],
    # [x, y difference between green and blue channels]

'''for i in circles_red[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)

for i in circles_green[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

for i in circles_blue[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)'''

pre_map = np.array(rgb_out)
with open('pre_map_dump.txt', 'w+') as fp:
    for e in pre_map:
        fp.write(str(e) + "\n")

for e in pre_map.tolist():
    cv2.circle(img, e[0], 35, (255, 0, 0), 2)

cv2.imshow('All circles', cv2.resize(img, (1920, 1080)))
cv2.waitKey(0)


def flow_map(patch_grid, image_size, pre_map):
    patch_num_x, patch_num_y = patch_grid
    patches = []
    for i in range(patch_num_x):
        for j in range(patch_num_y):
            x_from = i * image_size[0] // patch_num_x
            x_to = (i + 1) * image_size[0] // patch_num_x
            y_from = j * image_size[1] // patch_num_y
            y_to = (j + 1) * image_size[1] // patch_num_y
            patches.append(get_patch(x_from, x_to, y_from, y_to, pre_map))
    return patches


def get_patch(from_x, to_x, from_y, to_y, pre_map):
    m1 = pre_map[np.logical_and(pre_map[:, 0, 0] > from_x, pre_map[:, 0, 1] > from_y)]
    patch_circles = m1[np.logical_and(m1[:, 0, 0] < to_x, m1[:, 0, 1] < to_y)]
    # print(patch_circles)
    if patch_circles.tolist():
        ret = np.nanmedian(patch_circles[:, 1:, :], axis=0)
    else:
        '''print(from_x, to_x, from_y, to_y)
        print(m1.shape, patch_circles.shape)'''

        return []
    return ret.tolist()


print(flow_map((5, 5), img.shape[:2][::-1], pre_map))

