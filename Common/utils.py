"""
Common utility functions
"""
# Third-party library
import os
import math
import cv2.cv2 as cv2
import pandas as pd
from PyQt5 import QtGui
from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtGui import QPixmap, QImage


def make_dir(file_name):
    save_root = os.path.join(os.getcwd(), file_name)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    return save_root


def regex_init(regular_expression):
    regex = QRegExp(regular_expression)
    regex_instance = QtGui.QRegExpValidator()
    regex_instance.setRegExp(regex)
    return regex_instance


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


def scale_image(src, scale):
    return cv2.resize(src, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def read_csv(file_path):
    data = pd.read_csv(file_path)
    data = data.loc[2:]
    relative_coordinate = []
    for index, item in data.iterrows():
        relative_coordinate.append(eval(item["guideline"])["MIDDLE"])
    return relative_coordinate


# TODO: 后期模板比对逻辑需要基于实际模型来进行优化
def compare(root_path, src_file_name, target_file_name):
    error_index = []
    try:
        with open(os.path.join(root_path, src_file_name), "r") as template:
            src_data = template.read()
        with open(os.path.join(root_path, target_file_name), "r") as template:
            target_data = template.read()
    except Exception as error:
        return False, error_index, "读取文件错误, %s" % error
    if len(src_data) < len(target_data):
        return False, error_index, "两次编制针数不一致"
    for i in range(len(target_data)):
        if src_data[i] != target_data[i]:
            error_index.append(i)
    ratio = (len(target_data) - len(error_index)) * 100 // len(target_data)
    return True, error_index, ratio


# TODO test文件保存函数 后期删除
def save_image_by_needle(image, direction, col, row):
    default_path = r"C:\test\temp\D{}C{}".format(direction, col)
    if not os.path.exists(default_path):
        os.makedirs(default_path)
    cv2.imwrite(os.path.join(default_path, "R{}.bmp".format(row)), image)


def show_image_to_label(src, label, flag=None):
    frame_QPixmap = QPixmap.fromImage(QImage(cv2.cvtColor(src, flag) if flag else src, src.shape[1], src.shape[0], QImage.Format_RGB888))
    label.setPixmap(frame_QPixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))


def sort_file(file_path, target_row, mask_count):
    ordered_file_name = [0] * mask_count

    for root, dirs, files in os.walk(file_path):
        for file in files:
            _, row, index = file[:-4].split('_')
            if target_row in row and isinstance(ordered_file_name[int(index)], int):
                ordered_file_name[int(index)] = file
    return ordered_file_name


if __name__ == '__main__':
    print(sort_file(r"E:\clip\data", "398", 600))
