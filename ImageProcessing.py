import os
import time
import cv2.cv2 as cv2
import numpy as np
from math import fabs, sin, cos, radians

start = time.time()
src_image = cv2.imread("./R3152.jpg", cv2.IMREAD_GRAYSCALE)
end = time.time()
print(end - start)

start = time.time()
cv2.imwrite("./R3154.jpg", src_image)
end = time.time()

print(end - start)

# pt_a = (410, 160)
# pt_b = (441, 260)
# num = 0
#
# for root, dirs, files in os.walk(r"F:\learning\Data\AF00-0.9-N"):
#     for file in files:
#         num += 1
#         target = os.path.join(root, file)
#         src_image = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
#         print(src_image.shape)
#         # cv2.imshow('src', src_image)
#
#         height = src_image.shape[0]
#         width = src_image.shape[1]
#
#         degree = -124
#
#         heightNew = int(width * fabs(sin(radians(degree))) + height*fabs(cos(radians(degree))))
#         widthNew = int(height * fabs(sin(radians(degree))) + width*fabs(cos(radians(degree))))
#
#
#         transform_element = cv2.getRotationMatrix2D(center=(width / 2, height / 2), angle=degree, scale=1)
#         print(transform_element)
#
#         transform_element[0, 2] += (widthNew - width) / 2
#         transform_element[1, 2] += (heightNew - height) / 2
#         print(transform_element)
#
#         spin_image = cv2.warpAffine(src_image, transform_element, (widthNew, heightNew), borderValue=0)
#         # cv2.imshow('spin', spin_image)
#         # cv2.imwrite('./spin.jpg', spin_image)
#
#         target_region = spin_image[pt_a[1]:pt_b[1], pt_a[0]:pt_b[0]]
#         # cv2.imshow('target', target_region)
#
#         # 加入滤波和形态学运算会改善吗？
#
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3))
#
#         dilate_image = cv2.dilate(target_region, kernel)
#
#         # cv2.imshow('dilate', dilate_image)
#
#         canny_image = cv2.Canny(dilate_image, 50, 200)
#         #
#         # cv2.imshow('canny', canny_image)
#
#         lines = cv2.HoughLinesP(canny_image, rho=1, theta=np.pi / 180, threshold=30, minLineLength=30, maxLineGap=10)
#         rgb_image = cv2.cvtColor(target_region, cv2.COLOR_GRAY2RGB)
#         average_shift = 0
#         counts = 0
#         if type(lines) == np.ndarray and lines.shape[0] > 0:
#             for line in lines:
#                 line = line[0]
#
#                 # TODO：过滤水平直线
#                 if abs(line[0] - line[2]) < 10:
#                     cv2.line(rgb_image, (line[0], line[1]), (line[2], line[3]), color=(0, 255, 0))
#                     average_shift += line[0] + line[2]
#                     counts += 2
#             if counts:
#                 average_shift /= counts
#             else:
#                 average_shift += 300
#             print(average_shift)
#         else:
#             average_shift = 300
#
#         start_point_x = int(pt_a[0] + average_shift)
#         start_point_y = pt_a[1]
#         width = 31
#
#         spin_image = cv2.cvtColor(spin_image, cv2.COLOR_GRAY2RGB)
#         print(rgb_image.shape)
#         print(spin_image.shape)
#         spin_image[0:100, 0:31] = rgb_image
#         for i in range(4):
#             cv2.rectangle(img=spin_image, pt1=(start_point_x, start_point_y),
#                           pt2=(start_point_x + width * (i + 1), start_point_y + 100), color=(255, 100, 100), thickness=1)
#         #
#         # cv2.imshow('cut', spin_image)
#         # cv2.imshow('rgb', rgb_image)
#
#
#         cv2.imwrite('./imagecut/test-{}.jpg'.format(num), spin_image)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
