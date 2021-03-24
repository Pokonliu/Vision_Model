"""
Common utility functions
"""
import os
import re
import math
import cv2.cv2 as cv2
import pandas as pd


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


def rotate_image(src, angle):
    height, width = src.shape[:2]
    center_X, center_Y = width / 2, height / 2

    transform_element = cv2.getRotationMatrix2D(center=(center_X, center_Y), angle=angle, scale=1)
    cos = math.fabs(transform_element[0, 0])
    sin = math.fabs(transform_element[0, 1])

    height_new = int(width * sin + height * cos)
    width_new = int(height * sin + width * cos)

    transform_element[0, 2] += width_new / 2 - center_X
    transform_element[1, 2] += height_new / 2 - center_Y

    return cv2.warpAffine(src, M=transform_element, dsize=(width_new, height_new), borderValue=0)


def read_csv(file_path):
    data = pd.read_csv(file_path)
    data = data.loc[2:]
    relative_coordinate = []
    for index, item in data.iterrows():
        relative_coordinate.append(eval(item["mid point"])["MIDDLE"])
    return relative_coordinate


# TODO test文件保存函数 后期删除
def save_image_by_needle(image, direction, col, row):
    default_path = r"C:\test\temp\D{}C{}".format(direction, col)
    if not os.path.exists(default_path):
        os.makedirs(default_path)
    cv2.imwrite(os.path.join(default_path, "R{}.bmp".format(row)), image)


if __name__ == '__main__':
    print(read_csv("./final_坐标模板.csv"))