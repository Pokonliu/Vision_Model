import os
import numpy as np
import cv2.cv2 as cv2
import utils
import pandas
import math
import random


ANGLE = 56.5
start_coordinate = [454, 716]
width = 31
length = 74


def get_coordinate(src, angle):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("当前坐标：({},{})".format(x, y))
            coordinates.append([x, y])
            cv2.circle(rotate_img, (x, y), 1, (255, 0, 255), 1)
            cv2.putText(rotate_img, "({},{})".format(x, y), (x + 4, y - 4), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        cv2.imshow("Calibration", rotate_img)
    coordinates = []
    rotate_img = utils.rotate_image(src, angle)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    cv2.imshow("Calibration", rotate_img)

    key = cv2.waitKey(0)
    # 找起始
    if key == 13:
        print(coordinates)
        return True, coordinates[0]
    # 跳过
    if key == 27:
        return False, None, None


def get_offset_table(path):
    result = {}
    data = pandas.read_csv(path)
    for index, item in data.iterrows():
        result[item["file_name"]] = item["x_offset"]
    return result


def get_group_image(root_path, start_index, cycle, line_num, direction):
    """

    :param root_path:
    :param start_index:
    :param cycle:
    :param line_num:
    :param direction:
    :return:
    """
    result = {}
    total_file_set = set(os.listdir(root_path))
    # 过滤掉头照片
    discard_file_set = set(x for x in total_file_set if "DTR" in x or "DTL" in x)
    valid_file_set = total_file_set.difference(discard_file_set)
    print(len(valid_file_set))

    for i in range(len(valid_file_set) // 2):
        dir_name = "D{}C".format(direction) + str(start_index)
        dir_path = os.path.join(root_path, dir_name)
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                file_name = "R{}.bmp".format(str(line_num))
                if file_name in files:
                    result[dir_name] = cv2.imread(os.path.join(root, file_name), cv2.IMREAD_COLOR)
        else:
            print("{} not exist, please check original data".format(dir_name))
        start_index += cycle if direction == "L" else -cycle
    print(result.keys())
    return result


# 读取一行数据并进行拼接API
def stitching_picture(group_of_picture: dict, direction, start_coordinates, offset_table, angle, width, length):
    """
    通过一组图片集合来拼接出一张无重复图片
    :param group_of_picture: eg: {序号: 图片, 序号: 图片, .....]
    :param direction: 'R' or 'L'
    :param start_coordinates: (x, y)
    :param offset_table: [csv中读出的offset数据]
    :param angle: 旋转角度
    :param width: 针格像素宽度
    :param length: 针格像素高度
    :return:
    """
    # 初始化最终输出的图片用于堆叠
    target_image = None
    print(start_coordinates)
    for index, image in group_of_picture.items():
        offset = offset_table[index]
        rotated_img = utils.rotate_image(image, angle)
        count = offset / width

        choose = random.random() > 0.23
        print("当前概率：", choose)
        # True的概率大，False的概率小

        cropped_count = math.ceil(count) if choose else math.floor(count)
        next_offset = cropped_count - count if choose else count - cropped_count
        print("截取数量{}， 真实数量{}， 下一次偏移{}".format(cropped_count, count, next_offset))
        if "L" == direction:
            print(start_coordinates)
            print("A", start_coordinates[1], start_coordinates[1] + length, start_coordinates[0] - width * cropped_count, start_coordinates[0])
            cropped_img = rotated_img[start_coordinates[1]: start_coordinates[1] + length, start_coordinates[0] - width * cropped_count: start_coordinates[0]]
            start_coordinates[0] += int(-next_offset * width if choose else next_offset * width)
        else:
            cropped_img = rotated_img[start_coordinates[1]: start_coordinates[1] + length, start_coordinates[0]: start_coordinates[0] + width * cropped_count]
            start_coordinates[0] += int(next_offset * width if choose else - next_offset * width)
        if type(target_image) == np.ndarray:
            target_image = np.concatenate((cropped_img, target_image), axis=1)
        else:
            target_image = cropped_img
    print(target_image.shape)
    cv2.imwrite('./result.jpg', target_image)
    cv2.imshow('result', target_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return target_image


if __name__ == '__main__':
    left_offset_table = get_offset_table("./Template/left.csv")
    right_offset_table = get_offset_table("./Template/right.csv")
    image_table = get_group_image(root_path=r"F:\learning\Data\t\temp", start_index=-2104, cycle=8, line_num=928, direction="L")
    invalid_image_list = []
    start_coordinate = None
    for dir_name, img in image_table.items():
        res, coordinate = get_coordinate(img, ANGLE)
        if not res:
            invalid_image_list.append(dir_name)
        else:
            start_coordinate = coordinate
            break
    for item in invalid_image_list:
        image_table.pop(item)

    stitching_picture(group_of_picture=image_table, direction="L", start_coordinates=start_coordinate, offset_table=left_offset_table, angle=ANGLE, width=width, length=length)

