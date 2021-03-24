"""
Common utility functions
"""
import os
import cv2.cv2 as cv2
import time
import numpy as np


def split_hex(hex_str):
    _, result = hex_str.split('x')
    return result


def colourful_text(data, color):
    return "{}{}{}".format("<font color='{}'>".format(color), data, "<font>")


def filling(src, max_length, filler='0'):
    if len(src) < max_length:
        return (max_length - len(src)) * filler + src
    else:
        return src


def lowercase_to_uppercase(lineEdit):
    lineEdit.setText(lineEdit.text().upper())


# TODO test文件保存函数 后期删除
def save_image_by_needle(image, direction, col, row):
    default_path = r"C:\test\temp\D{}C{}".format(direction, col)
    if not os.path.exists(default_path):
        os.makedirs(default_path)
    cv2.imwrite(os.path.join(default_path, "R{}.bmp".format(row)), image)


# if __name__ == '__main__':
#     src = cv2.imread('./R3152.jpg', cv2.IMREAD_GRAYSCALE)
#
#     io_time = 0
#     a = time.time()
#     for i in range(10000):
#         default_path = r"C:\test\temp2\D{}C{}".format('T', str(i))
#         if not os.path.exists(default_path):
#             os.makedirs(default_path)
#         tar = os.path.join(default_path, "R{}.bmp".format('1'))
#         start = time.time()
#         cv2.imwrite(tar, src)
#         end = time.time()
#         io_time += (end - start)
#     b = time.time()
#     print(io_time)
#     print(b - a)
