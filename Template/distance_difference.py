# 此程序是计算两帧图像之间移动了多少像素， 可以自动制作相对位移模板

import cv2
import os
import numpy as np
import math
import pandas
import serial

# 旋转角度
angle = 126.707959

# 设置每张模板图的x方向起始坐标
template_x = 284

image_folder_path = r'J:\data0610\temp'

# 保存异常数据的txt文件路径
l_txt = r'G:\test\左侧偏移异常数据.txt'
r_txt = r'G:\test\右侧偏移异常数据.txt'

# # 相对移动距离坐标模板csv
left_csv_path = r'C:\Users\S.s\Desktop\making_new_model(1)\relative_left.csv'
right_csv_path = r'C:\Users\S.s\Desktop\making_new_model(1)\relative_right.csv'


# 初始化csv文件，保存相对坐标模板
def init_csv(path):
    try:
        # 如果path已经存在
        if os.path.exists(path):
            # 读取csv文件
            data = pandas.read_csv(path)
            # 如果长度小于1 证明是空
            if len(data) < 1:
                return True, None
            # 否则，
            else:
                # 显示csv文件的最后一行
                bottom = data.tail(1)
                return True, bottom["file_name"].values[0]
        else:
            # 制作表头
            data = pandas.DataFrame(columns=["file_name", "distance"])
            data.to_csv(path, mode="a", index=False, header=True)
    except Exception as error:
        print(error)


# 获取图像总目录下规定的文件,向左和向右的文件夹名称分别保存在两个列表中
def get_folder_names(root_path):
    dlc_folder = []
    drc_folder = []
    folders = os.listdir(root_path)
    for name in folders:
        if name[:3] == "DLC" and 142 >= int(name[3:]) >= -217:
            dlc_folder.append(name)
            dlc_folder.sort(key=lambda x: int(x[3:]))
        elif name[:3] == "DRC" and 141 >= int(name[3:]) >= -218:
            drc_folder.append(name)
            drc_folder.sort(key=lambda x: int(x[3:]))
    drc_folder.reverse()
    return dlc_folder, drc_folder


# 获取每个图像文件夹下图像名称，为保证能全部获取,遍历所有文件夹
def get_row_name(root_path):
    # 记数
    l_number, r_number = 1, 1
    # 保存行号
    l_all_row_names, r_all_row_names = [], []
    # 缓存
    temp1, temp2 = [], []
    #
    l_new_row_list, r_new_row_list = [], []

    l_folder_names, r_folder_names = get_folder_names(root_path)
    # 获取左侧所有行的名称
    for l_folder_name in l_folder_names:
        row_names = set(os.listdir(os.path.join(root_path, l_folder_name)))
        l_all_row_names.append(row_names)
        if l_number >= 2:
            temp = l_all_row_names[0].union(l_all_row_names[1])
            l_all_row_names[0] = temp
            l_all_row_names.pop()

        if l_number == len(l_folder_names):
            l_all_row_names = list(l_all_row_names[0])
        l_number += 1

    # 按行号排序
    for l_row_name in l_all_row_names:
        row_number = l_row_name.split(sep='.')[0]
        temp1.append(row_number)
    temp1.sort(key=lambda x: int(x[1:]))
    for name in temp1:
        l_new_row_list.append(name + '.bmp')

    # 获取右侧所有行的名称
    for r_folder_name in r_folder_names:
        row_names = set(os.listdir(os.path.join(root_path, r_folder_name)))
        r_all_row_names.append(row_names)
        if r_number >= 2:
            temp = r_all_row_names[0].union(r_all_row_names[1])
            r_all_row_names[0] = temp
            r_all_row_names.pop()

        if r_number == len(r_folder_names):
            r_all_row_names = list(r_all_row_names[0])
        r_number += 1

    # 按行号排序
    for r_row_name in r_all_row_names:
        row_number = r_row_name.split(sep='.')[0]
        temp2.append(row_number)
    temp2.sort(key=lambda x: int(x[1:]))
    for name in temp2:
        r_new_row_list.append(name + '.bmp')

    return l_new_row_list, r_new_row_list


#  旋转图像
def rotateScaleImg(img, angle):
    # image.shape得到的是图像的高（row）、宽（col）、通道-->(rows, cols, channels)，
    # [1::-1] 1表示从元组中的第二个值开始，空表示到末尾，-1表示倒序，间隔为 1.这样写是从第二个值开始到开头
    # /2是np中的广播机制，宽、高全部除以2
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    W, H = img.shape[1::-1]

    # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    rotate_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)

    # 计算旋转后输出图形的尺寸
    rotated_width = math.ceil(H * math.fabs(rotate_matrix[0, 1]) + W * math.fabs(rotate_matrix[1, 1]))
    rotated_height = math.ceil(W * math.fabs(rotate_matrix[0, 1]) + H * math.fabs(rotate_matrix[1, 1]))

    # 防止切边，对平移矩阵B进行修改
    rotate_matrix[0, 2] += (rotated_width - W) / 2
    rotate_matrix[1, 2] += (rotated_height - H) / 2

    # 应用旋转
    img_rotated = cv2.warpAffine(img, rotate_matrix, (rotated_width, rotated_height))
    return img_rotated


# # 将文件夹按列数字排序
# def sort_folder(root_path):
#     temp = []
#     new_row_list = []
#     row_list = get_row_name(root_path)
#     for row_name in row_list:
#         row_number = row_name.split(sep='.')[0]
#         temp.append(row_number)
#     temp.sort(key=lambda x: int(x[1:]))
#     for name in temp:
#         new_row_list.append(name + '.bmp')
#     return new_row_list


# 获取模板图像， 此处没用到
def get_template_image(path):
    tem_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    template_image = rotateScaleImg(tem_img, angle)[326: 544, 284:507]
    return template_image


# 三种模板匹配方式:对比使用cv2.TM_SQDIFF_NORMED较为稳定  ***测试用***
# def template_demo(template_image, detection_image):
#     # 各种匹配算法
#     methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
#     # 获取模板图像的高宽
#     th, tw = template_image.shape[:2]
#     for md in methods:
#         result = cv2.matchTemplate(detection_image, template_image, md)
#
#         # result是我们各种算法下匹配后的图像
#         # cv2.imshow("%s"%md,result)
#         # cv2.waitKey(0)
#
#         # 获取的是每种公式中计算出来的值，每个像素点都对应一个值
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#         print(min_val, max_val, min_loc, max_loc)
#         if md == cv2.TM_SQDIFF_NORMED:
#             tl = min_loc  # tl是左上角点
#             print(tl[0], tl[1])
#         else:
#             tl = max_loc
#             print(tl[0], tl[1])
#         br = (tl[0] + tw, tl[1] + th)  # 右下点
#         # 画矩形
#         cv2.rectangle(detection_image, tl, br, (0, 0, 255), 2)
#         cv2.imshow("match-%s" % md, detection_image)
#         cv2.waitKey(0)


# 计算相邻两帧的图像的相对移动距离 （pixel）
def calculate_moving_distance(template_image, target_image):
    """
    :param template_image: 每次检测需要的模板图像，从上一帧指定区域选出
    :param target_image: 当前帧图像
    :return: 返回的是当前帧相比较之前帧移动了多少个像素
    """
    tem_img_h, tem_img_w = template_image.shape[:2]
    # result模板匹配后得到的图像
    result = cv2.matchTemplate(target_image, template_image, cv2.TM_SQDIFF_NORMED)
    # # 获取的是每种公式中计算出来的值，每个像素点都对应一个值， cv2.TM_SQDIFF_NORMED求出对应的坐标是最小值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    point1 = min_loc
    moving_distance = point1[0] - template_x
    # 在目标图上绘制检测结果
    # cv2.rectangle(target_image, point1, (point1[0] + tem_img_w, point1[1] + tem_img_h), (255, 255, 255), 2)
    # cv2.imshow("result", target_image)
    # cv2.waitKey(0)
    # cv2.destroyWindow('result')

    return abs(moving_distance)


# 把每一行的图像都保存起来
def get_every_row(row_name, diretion):
    image_one_row = {}
    # 获取所有文件夹中列的列表
    dlc, drc = get_folder_names(image_folder_path)
    if diretion == "L":
        for l_folder in dlc:
            image_path = os.path.join(image_folder_path, l_folder, row_name)
            if os.path.exists(image_path):
                images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image_one_row[l_folder] = images
            else:
                continue
    elif diretion == 'R':
        for r_folder in drc:
            image_path = os.path.join(image_folder_path, r_folder, row_name)
            if os.path.exists(image_path):
                images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image_one_row[r_folder] = images
            else:
                continue
    return image_one_row


# 保存相对漂移量数据
def save_relative_data(csv_path, col_name, dst):
    init_csv(csv_path)
    relative_distance = {"file_name": col_name,
                         "distance": dst}
    df = pandas.DataFrame(relative_distance,
                          columns=["file_name", "distance"], index=[0])
    df.to_csv(csv_path, mode="a", index=False, header=False)


# 此处检测当前帧相对于之前帧移动了多少距离，同时可以指定特定行，制作相对坐标模板
def start_detection(direction, l_folder, r_folder):
    if direction == "L":
        for l_row in l_rows:
            with open(l_txt, mode='a+') as l_file_handle:
                for i in range(len(l_folder) - 1):
                    if i == 0:
                        distance = 0
                    else:
                        path_temp = os.path.join(image_folder_path, l_folder[i - 1], l_row)
                        path_target = os.path.join(image_folder_path, l_folder[i], l_row)
                        if not os.path.exists(path_temp) or not os.path.exists(path_target):
                            continue
                        tem_img = rotateScaleImg(cv2.imread(path_temp), angle)[326: 544, template_x: template_x + 323]
                        # cv2.imshow("t", tem_img)
                        # cv2.waitKey(0)
                        # cv2.destroyWindow('t')

                        target_img = rotateScaleImg(cv2.imread(path_target), angle)
                        distance = calculate_moving_distance(tem_img, target_img)
                        # print('{}行中{}列移动距离为{}  \n'.format(l_row.split(sep='.')[0], l_folder[i], distance))

                        if distance >= 60 or distance <= 49:
                            print('{}行中{}列移动距离为{}  \n'.format(l_row.split(sep='.')[0], l_folder[i], distance))
                            msg = l_row.split(sep='.')[0] + ' 行 ' + l_folder[i] + ' 列 ' + ' 移动距离为：' \
                                  + str(distance) + '  超出范围' + '\n'
                            l_file_handle.write(msg)

                    # 此处是选取每个文件夹下都有的图片，制作相对位移模板
                    # if l_row == 'R6.bmp':
                    #     save_relative_data(left_csv_path, l_folder[i], distance)
            l_file_handle.close()

    elif direction == 'R':
        with open(r_txt, mode='a+') as r_file_handle:
            for r_row in r_rows:
                for i in range(len(r_folder) - 1):
                    if i == 0:
                        distance = 0
                    else:
                        path_temp = os.path.join(image_folder_path, r_folder[i - 1], r_row)
                        path_target = os.path.join(image_folder_path, r_folder[i], r_row)
                        if not os.path.exists(path_temp) or not os.path.exists(path_target):
                            continue
                        tem_img = rotateScaleImg(cv2.imread(path_temp), angle)[326: 544, template_x: template_x + 323]
                        # cv2.imshow("t", tem_img)
                        # cv2.waitKey(0)

                        target_img = rotateScaleImg(cv2.imread(path_target), angle)
                        distance = calculate_moving_distance(tem_img, target_img)
                        # print('{}行中{}列移动距离为{}  \n'.format(r_row.split(sep='.')[0], r_folder[i], distance))

                        if distance >= 60 or distance <= 50:
                            print('{}行中{}列移动距离为{}  \n'.format(r_row.split(sep='.')[0], r_folder[i], distance))

                            msg = r_row.split(sep='.')[0] + ' 行 ' + r_folder[i] + ' 列 ' + ' 移动距离为：' \
                                  + str(distance) + '  超出范围' + '\n'
                            r_file_handle.write(msg)

                    # # 保存数据
                    # if r_row == 'R1.bmp':
                    #     save_relative_data(right_csv_path, r_folder[i], distance)

        r_file_handle.close()


if __name__ == "__main__":
    # 获取不同方向行名称list
    l_rows, r_rows = get_row_name(image_folder_path)
    l_folders, r_folders = get_folder_names(image_folder_path)
    # directions = ['L', 'R']
    # for direction in directions:
    #     start_detection(direction)
    directions = input("请输入要检测的方向： L or R \n")
    start_detection(directions, l_folders, r_folders)
    print("*" * 8 + 'finished!' + "*" * 8)
