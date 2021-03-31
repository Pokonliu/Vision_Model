import os
import numpy as np
import cv2.cv2 as cv2
import utils
import pandas
import math


ANGLE = 56.5
start_coordinate = [454, 716]
width = 31
length = 74


def get_coordinate(path, angle):
    def mouse_callback(event, x, y, flags, param):
        nonlocal effectiveness
        if event == cv2.EVENT_LBUTTONDOWN:
            print("当前坐标：({},{})".format(x, y))
            coordinates.append([x, y])
            cv2.circle(rotate_img, (x, y), 1, (255, 0, 255), 1)
            cv2.putText(rotate_img, "({},{})".format(x, y), (x + 4, y - 4), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        if event == cv2.EVENT_RBUTTONDOWN:
            effectiveness = "invalid"
            cv2.putText(rotate_img, "invalid", (x + 4, y - 4), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        cv2.imshow("Calibration", rotate_img)
    coordinates = []
    effectiveness = "valid"
    src = cv2.imread(path, cv2.IMREAD_COLOR)
    print(path)
    rotate_img = utils.rotate_image(src, angle)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    cv2.imshow("Calibration", rotate_img)

    key = cv2.waitKey(0)
    # RUNNING
    if key == 13:
        print(coordinates, effectiveness)
        return True, coordinates[:2], effectiveness
    # QUIT
    if key == 27:
        return False, None, None



file_224 = r"F:\learning\Data\t\temp\DLC224\R926.bmp"
file_232 = r"F:\learning\Data\t\temp\DLC232\R926.bmp"
file_240 = r"F:\learning\Data\t\temp\DLC240\R926.bmp"
file_248 = r"F:\learning\Data\t\temp\DLC248\R926.bmp"
file_256 = r"F:\learning\Data\t\temp\DLC256\R926.bmp"
# get_coordinate(file_224, ANGLE)

hash_table = {"DLC224": file_224,
              "DLC232": file_232,
              "DLC240": file_240,
              "DLC248": file_248,
              "DLC256": file_256}

data = pandas.read_csv("./left.csv")
print(data.head(5))

offset_table = {}

for index, item in data.iterrows():
    offset_table[item["file_name"]] = item["x_offset"]

image = cv2.imread(r"F:\learning\Data\t\temp\DLC216\R926.bmp", cv2.IMREAD_COLOR)
r_img = utils.rotate_image(image, ANGLE)
cropped_img = r_img[start_coordinate[1]: start_coordinate[1] + length, start_coordinate[0]: start_coordinate[0] + 1]

target_image = cropped_img

for k, v in hash_table.items():
    print("当前图片:", k)
    image = cv2.imread(v, cv2.IMREAD_COLOR)
    r_img = utils.rotate_image(image, ANGLE)

    offset = offset_table[k]
    print("偏移量:", offset)

    count = offset / width
    print("偏移个数:", count)

    # 取上整
    cropped_count = math.ceil(count)
    print("偏移个数取整:", cropped_count)

    next_offset = cropped_count - count
    print("下次需要移动距离:", next_offset)

    print("本次起点坐标:", start_coordinate)
    cropped_img = r_img[start_coordinate[1]: start_coordinate[1] + length, start_coordinate[0] - width * cropped_count: start_coordinate[0]]
    target_image = np.concatenate((cropped_img, target_image), axis=1)
    print("当前图片形状:", target_image.shape)

    # 对下一次坐标修正
    start_coordinate[0] -= int(next_offset * width)
    print("下次起点坐标:", start_coordinate)

    cv2.imshow("d", r_img)
    cv2.imshow("c", cropped_img)
    cv2.imshow("T", target_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

