import csv

import cv2
import math
import numpy as np
import os
import pandas

# 模板中一个针格的像素数
stitch_model = 31.6
# 新图像中一个针格的像素数
stitch_now = 31.6
# 图像放缩系数
scale = stitch_model / stitch_now
# 向左的起点col
start_col_l = 'DLC0'
# 向右的起点col
start_col_r = 'DRC0'

# 旋转的角度
angle = 126.707959

# 被截图的图片路径
image_root_path1 = r'J:\temp0527-1'
# 获取绝对坐标的0号图像的路径
l_image = r'J:\temp0527-1\DLC0\R236.bmp'
r_image = r'J:\temp0527-1\DRC0\R237.bmp'

# 获取行信息的文件夹路径
dlc0 = r'J:\temp0527-1\DLC0'
drc0 = r'J:\temp0527-1\DRC0'
# 保存截图的路径
saveImage_root_path = r'G:\testcode\data0609'
# csv文件路径
# 相对坐标
l_relative_path = r'C:\Users\S.s\Desktop\making_new_model(1)\坐标模板备份\relative_left.csv'
r_relative_path = r'C:\Users\S.s\Desktop\making_new_model(1)\坐标模板备份\relative_right.csv'
# 绝对坐标
l_absolute_path = r'C:\Users\S.s\Desktop\making_new_model(1)\坐标模板备份\l_absolute_coordinate.csv'
r_absolute_path = r'C:\Users\S.s\Desktop\making_new_model(1)\坐标模板备份\r_absolute_coordinate.csv'


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
            data = pandas.DataFrame(columns=["file_name", "ab_distance", 'number', 'index'])
            data.to_csv(path, mode="a", index=False, header=True)
    except Exception as error:
        print(error)


#  旋转图像
def rotateScaleImg(img, angle, scale=1.):
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
    img_rotated_resized = cv2.resize(img_rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img_rotated


# 获取初始绝对坐标
def get_starting_point(direction):
    # 鼠标点击事件
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("当前坐标：({},{})".format(x, y))
            start.append([x, y])
            cv2.circle(rotate_img, (x, y), 2, (255, 0, 255), 1)
            cv2.putText(rotate_img, "({},{})".format(x, y), (x + 4, y - 4), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255, 0, 255),
                        1)
        if event == cv2.EVENT_RBUTTONDOWN:
            effectiveness = "invalid"
            cv2.putText(rotate_img, "invalid", (x + 40, y - 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        cv2.imshow('rotate', rotate_img)

    start = []
    start_img = None

    if direction == 'L':
        start_img = cv2.imread(l_image)
    elif direction == "R":
        start_img = cv2.imread(r_image)

    rotate_img = rotateScaleImg(start_img, angle)
    cv2.namedWindow('rotate')
    cv2.setMouseCallback('rotate', mouse_callback)
    cv2.imshow('rotate', rotate_img)
    cv2.waitKey(0)
    cv2.destroyWindow('rotate')

    print(start[-1][0])
    return start[-1][0]


# 通过检测优化起始切割坐标点
def optimize_start_point(image_path, start_point):
    points = []
    point_i = 0
    src_img = cv2.imread(image_path)
    rot_img = rotateScaleImg(src_img, angle)
    # 进行Hough_line直线检测
    section_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)[100: 500, start_point - 16: start_point + 16]
    # cv2.imshow('a', section_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('a')

    # 滤波操作
    section_img = cv2.GaussianBlur(section_img, (3, 3), 0)
    edges = cv2.Canny(section_img, 50, 200, apertureSize=3)

    # 形态学操作，把断点连起来形成直线
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    # cv2.imshow('erosion', erosion)
    # cv2.waitKey(0)
    # cv2.destroyWindow('erosion')

    # 检测直线
    lines = cv2.HoughLinesP(erosion, 1, np.pi / 180, 80, 110, 10)
    if lines is None:
        return start_point
    else:
        # 遍历每一条直线
        for i in range(len(lines)):
            cv2.line(src_img, (lines[i, 0, 0] + (start_point - 12), lines[i, 0, 1] + 100),
                     (lines[i, 0, 2] + (start_point - 16), lines[i, 0, 3] + 100), (255, 0, 255), 1)
            # cv2.imshow("result", img)
            # cv2.waitKey(0)
            # cv2.destroyWindow('result')
            points.append(lines[i][0][0])
        # 计算直线x方向上起点坐标的平均值
        for j in range(len(points)):
            point_i += points[j]
        # 将坐标还原到原图上 template_x - y   y根据上方截取roi区域的坐标来修改
        point = (round(point_i // len(points))) + (start_point - 16)
        # 使用做小x坐标作为切割起始坐标
        # point = min(points) + (template_x - 16)
        # print('初始化的坐标', start_point)
        # print('优化后的坐标', point)
        return int(point)


# 根据相对偏移量，推算出的绝对坐标，起始点为x轴坐标一个数值，标量
def next_point(last_point, shift_value, col, starting_point):
    """
    :param last_point: 上一帧中切割针格的起点坐标
    :param shift_value: 相对偏移量
    :param col: 列号
    :param starting_point: 手动确定的行号为0的绝对坐标
    :return:
    """
    point = 0
    # 范围
    limits = [starting_point - stitch_model, starting_point + stitch_model]
    num_stitches = 2
    if col[:3] == "DLC":
        if int(col[3:]) < 0:
            # 推算的当前图的切割起点
            # TODO 因为硬件信号错误导致信息与之前相反，方向相反，正负相反,后期要改回来
            cur_pt = last_point - (2 * stitch_model - shift_value)
            # 判断当前切割点不偏移出以0点为中心，左右两个针格的范围，确定切割几个针格
            num_stitches = 2 if limits[0] < cur_pt < limits[1] else 1
            # 确定当前切割起始点
            point = last_point - round(num_stitches * stitch_model - shift_value)
        else:
            # 推算的当前图的切割起点
            cur_pt = last_point + (2 * stitch_model - shift_value)
            # 判断当前切割点不偏移出以0点为中心，左右两个针格的范围，确定切割几个针格
            num_stitches = 2 if limits[0] < cur_pt < limits[1] else 1

            point = last_point + round(num_stitches * stitch_model - shift_value)
    elif col[:3] == "DRC":
        if int(col[3:]) > 0:
            # 推算的当前图的切割起点
            cur_pt = last_point + (2 * stitch_model - shift_value)
            # 判断当前切割点不偏移出以0点为中心，左右两个针格的范围，确定切割几个针格
            num_stitches = 2 if limits[0] < cur_pt < limits[1] else 1
            # 确定当前切割起始点
            point = last_point + round(num_stitches * stitch_model - shift_value)
        else:
            # 推算的当前图的切割起点
            cur_pt = last_point - (2 * stitch_model - shift_value)
            # 判断当前切割点不偏移出以0点为中心，左右两个针格的范围，确定切割几个针格
            num_stitches = 2 if limits[0] < cur_pt < limits[1] else 1
            # 确定当前切割起始点
            point = last_point - round(num_stitches * stitch_model - shift_value)

    return point, num_stitches


# 从相对坐标中获得绝对坐标
def calculate_absolute_coordinate(starting_col, x_0, relative_coordinate_path, row_name, absolute_path):
    """
    :param starting_col: 获取初始绝对坐标图像所在的列号
    :param x_0: 标定的初始的绝对坐标
    :param relative_coordinate_path:相对偏移量的csv路径
    :param row_name:行号，也就是图片名称
    :param absolute_path:保存绝对坐标的csv文件
    :return:
    """
    # 创建保存绝对坐标的csv文件
    init_csv(absolute_path)
    # 从csv文件读取相对坐标，保存在list中
    relative_coordinates = []
    # 获取所有的列号信息
    cols = []
    # 创建字典，保存每行的绝对坐标和切割个数
    absolute_coordinates = {}
    # 初始化标定图像列号在csv中的索引
    marked_index = 0
    

    # 读取相对坐标的csv
    csv_folder = open(relative_coordinate_path)
    csv_reader = csv.reader(csv_folder)
    # 获得所有的列号和相对偏移量
    for row in csv_reader:
        if csv_reader.line_num >= 2:
            cols.append(row[0])
            relative_coordinates.append(eval(row[1]))

    # 得到标定图像在csv中的索引
    for index, name in enumerate(cols):
        if name == starting_col:
            marked_index = index

    # 从start开始向前的cols
    forward_list = cols[:marked_index + 1]
    forward_list.reverse()
    # 从start开始向后的cols
    back_list = cols[marked_index:]
    # 向前计算各个col的绝对坐标
    for i in range(len(forward_list)):
        # 如果是第一个，就是起始dlc/drc-0
        if i == 0:
            # 保存对应的名称和鼠标点击得到的绝对坐标,切割的针格个数
            absolute_coordinates = {starting_col: [x_0, 2]}
        # 其他需要推算的坐标
        else:
            # 推算出截取点和每个图截取几个针格
            point, number = next_point(absolute_coordinates[forward_list[i - 1]][0],
                                       relative_coordinates[marked_index - i + 1],
                                       forward_list[i], x_0)
            image_name = row_name + '.bmp'
            img_path = os.path.join(image_root_path1, forward_list[i], image_name)
            # 如果某col文件夹中读取当前名称的图像不存在，就按计算的值保存
            if not os.path.exists(img_path):
                absolute_coordinates[forward_list[i]] = [point, number]
                continue
            # 获取当前图像，检测边缘，优化截取针格的起始点
            optimize_point = optimize_start_point(img_path, point)
            # 保存
            absolute_coordinates[forward_list[i]] = [optimize_point, number]
    # 向后计算
    for i in range(len(back_list)):
        if i == 1:
            image_name = row_name + '.bmp'
            point, number = next_point(absolute_coordinates[starting_col][0],
                                       relative_coordinates[marked_index + i], back_list[i], x_0)
            img_path = os.path.join(image_root_path1, back_list[i], image_name)
            # 如果某col文件夹中读取当前名称的图像不存在，就按计算的值保存
            if not os.path.exists(img_path):
                absolute_coordinates[back_list[i]] = [point, number]
                continue
            optimize_point = optimize_start_point(img_path, point)
            absolute_coordinates[back_list[i]] = [optimize_point, number]
        elif i > 1:
            image_name = row_name + '.bmp'
            point, number = next_point(absolute_coordinates[back_list[i - 1]][0],
                                       relative_coordinates[marked_index + i],
                                       back_list[i], x_0)
            img_path = os.path.join(image_root_path1, back_list[i], image_name)
            # 如果某col文件夹中读取当前名称的图像不存在，就按计算的值保存
            if not os.path.exists(img_path):
                absolute_coordinates[back_list[i]] = [point, number]
                continue
            optimize_point = optimize_start_point(img_path, point)
            absolute_coordinates[back_list[i]] = [optimize_point, number]
    return absolute_coordinates, cols


# 计算每幅图片切割针格的起始序号
def calculate_index(coordinate, col_name_list):
    index_list = {}
    start_index = 1
    for i, col_name in enumerate(col_name_list):
        index_list[col_name] = start_index
        start_index += int(coordinate[col_name][1])

    return index_list


# 保存计算得出的绝对路径
def save_absolute_coordinates(absolute_path, col_list, absolute_coordinate, stitch_index):
    init_csv(absolute_path)
    for col in col_list:
        # name = cols
        # coordinate = ab_coordinate
        temp = {"file_name": col,
                "ab_distance": absolute_coordinate[col][0],
                "number": absolute_coordinate[col][1],
                "index": stitch_index[col]}
        df = pandas.DataFrame(temp,
                              columns=["file_name", "ab_distance", "number", "index"], index=[0])
        df.to_csv(absolute_path, mode="a", index=False, header=False)


# 保存绝对坐标
def save_absolute_coordinate(direction):
    starting_point = get_starting_point(direction)
    if direction == 'L':
        # 计算出绝对坐标，以字典形式保存key：col名称， value：[ab_coord, number]
        abs_coordinates, cols_list = calculate_absolute_coordinate(start_col_l, starting_point,
                                                                   l_relative_path, 'R244', l_absolute_path)
        # 得到切割针格的序号起点
        stitch_indexs = calculate_index(abs_coordinates, cols_list)
        # 保存数据
        save_absolute_coordinates(l_absolute_path, cols_list, abs_coordinates, stitch_indexs)

    elif direction == 'R':
        # 计算出绝对坐标，以字典形式保存key：col名称， value：[ab_coord, number]
        abs_coordinates, cols_list = calculate_absolute_coordinate(start_col_r, starting_point,
                                                                   r_relative_path, 'R245', r_absolute_path)
        # 得到切割针格的序号起点
        stitch_indexs = calculate_index(abs_coordinates, cols_list)
        # 保存数据
        save_absolute_coordinates(r_absolute_path, cols_list, abs_coordinates, stitch_indexs)


# 切割
def clip_stitch_image(direction, col_names, absolute_coordinates, stitch_index, stitch_nums=None):
    """
    :param direction:  方向
    :param col_names: 每个方向列名称的列表
    :param absolute_coordinates: 计算得到的绝对坐标和切割的针格数
    :param stitch_nums: 可设置的人工认为需要截取针格的个数
    :param stitch_index: 序号
    :return:
    """
    # 获取所有的行的信息， DLC0/DRC0文件夹下的所有行信息即可
    l_row_name = os.listdir(dlc0)
    r_row_name = os.listdir(drc0)
    if direction == "L":
        # 遍历每个col文件夹下row_name的图像
        for row_name in l_row_name:
            # 创建存图的路径
            saveImage_path = os.path.join(saveImage_root_path, row_name)
            isExists = os.path.exists(saveImage_path)
            if not isExists:
                os.makedirs(saveImage_path)
            # col信息
            for col in col_names:
                image_path = os.path.join(image_root_path1, col, row_name)
                if not os.path.exists(image_path):
                    continue
                start_point = optimize_start_point(image_path, absolute_coordinates[col][0])
                src_img = rotateScaleImg(cv2.imread(image_path), angle)

                for i in range(absolute_coordinates[col][1]):
                    # TODO 由于硬件信号错误，方向相反，向左本应是减，但此处要改为加，后期要改回来
                    stitch_img = src_img[400:500,
                                 round(start_point + i * stitch_model)
                                 :round(start_point + (i + 1) * stitch_model - 1),
                                 :]
                    # if col == 'DLC31':
                    # cv2.imshow('a', stitch_img)
                    # cv2.waitKey(0)
                    # cv2.destroyWindow('a')
                    image_name = col + '_' + row_name.split('.')[0] + '_' + str(stitch_index[col] + i) + ".jpg"
                    save_path = os.path.join(saveImage_path, image_name)
                    cv2.imwrite(save_path, stitch_img)
    if direction == 'R':
        for row_name in r_row_name:
            # 创建存图的路径
            saveImage_path = os.path.join(saveImage_root_path, row_name)
            isExists = os.path.exists(saveImage_path)
            if not isExists:
                os.makedirs(saveImage_path)
            for col in col_names:
                image_path = os.path.join(image_root_path1, col, row_name)
                if not os.path.exists(image_path):
                    continue
                start_point = optimize_start_point(image_path, absolute_coordinates[col][0])
                src_img = rotateScaleImg(cv2.imread(image_path), angle)

                for i in range(absolute_coordinates[col][1]):
                    # TODO 由于硬件信号错误，方向相反，向右本应是加，但此处要改为减，后期要改回来
                    stitch_img = src_img[380:500,
                                 round(start_point - (i + 1) * stitch_model)
                                 :round(start_point - i * stitch_model),
                                 :]
                    # cv2.imshow('a', stitch_img)
                    # cv2.waitKey(0)
                    # cv2.destroyWindow('a')
                    image_name = col + '_' + row_name.split('.')[0] + '_' + str(stitch_index[col] + i) + ".jpg"
                    save_path = os.path.join(saveImage_path, image_name)
                    cv2.imwrite(save_path, stitch_img)


if __name__ == "__main__":
    abs_coordinates = {}
    cols_list = []
    directions = input("请输入方向：L or R  \n")
    # 获得初始的绝对坐标，选择序号为0的文件下的图像
    starting_point = get_starting_point(directions)
    # 计算出绝对坐标，以字典形式保存key：col名称， value：[ab_coord, number]
    if directions == 'L':
        abs_coordinates, cols_list = calculate_absolute_coordinate(start_col_l, starting_point,
                                                                   l_relative_path, 'R244', l_absolute_path)

    elif directions == 'R':
        abs_coordinates, cols_list = calculate_absolute_coordinate(start_col_r, starting_point,
                                                                   r_relative_path, 'R245', r_absolute_path)
    # 得到切割针格的序号起点
    stitch_indexs = calculate_index(abs_coordinates, cols_list)
    # 切图
    clip_stitch_image(directions, cols_list, abs_coordinates, stitch_indexs)
