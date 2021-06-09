import os
import pandas
import cv2.cv2 as cv2
import utils


ANGLE = 56.5
TARGET_PATH = r"F:\learning\Data\temp-1"

# 比例变换：174 pixel：1 cm：10 mm
# 针格的长度：1.81 mm: 31.494 pixel


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


def cal_offset(coordinate_list):
    x_offset = [0]
    for i in range(len(coordinate_list) - 1):
        x_offset.append(abs(coordinate_list[i + 1][1][0] - coordinate_list[i][0][0]))
    return x_offset


def calibration_images(csv_path, root_path, files, line_num, angle, direction):
    res, last_file_name = init_csv(csv_path)
    if res:
        search_flag = True if last_file_name else False
    else:
        print("初始化错误")
        return False

    for index, file_name in enumerate(files):
        # 跳过之前制作的图片
        if search_flag and last_file_name != file_name:
            continue
        elif last_file_name == file_name:
            search_flag = False
            continue

        file_path = os.path.join(root_path, file_name, "R{}.bmp".format(str(line_num)))
        res, coordinate, effectiveness = get_coordinate(file_path, angle)
        if not res:
            print("退出制作")
            return True
        # 从第二幅图开始制作offset
        x_axis_offset = 0
        if index > 0:
            last_start_point = pandas.read_csv(csv_path).tail(1)['start_point'].values[0]
            x_axis_offset = abs(coordinate[1][0] - eval(last_start_point)[0])

        hash_table = {"file_name": file_name,
                      "start_point": str(coordinate[0]),
                      "end_point": str(coordinate[1]),
                      "x_offset": x_axis_offset,
                      "effectiveness": effectiveness
                      }
        df = pandas.DataFrame(hash_table, columns=["file_name", "start_point", "end_point", "x_offset", "effectiveness"], index=[0])
        df.to_csv(csv_path, mode="a", index=False, header=False)
    return True


def init_csv(path):
    try:
        if os.path.exists(path):
            data = pandas.read_csv(path)
            if len(data) < 1:
                return True, None
            else:
                bottom = data.tail(1)
                return True, bottom["file_name"].values[0]
        else:
            # 制作表头
            data = pandas.DataFrame(columns=["file_name", "start_point", "end_point", "x_offset", "effectiveness"])
            data.to_csv(path, mode="a", index=False, header=True)
            return True, None
    except Exception as error:
        print(error)
        return False, None


def make_template(root_path, start_index, cycle, angle, L_line, R_line):
    left_direction_file_set = []
    right_direction_file_set = []
    total_file_set = set(os.listdir(root_path))
    # 过滤掉头照片
    discard_file_set = set(x for x in total_file_set if "DTR" in x or "DTL" in x)
    valid_file_set = total_file_set.difference(discard_file_set)

    print(len(valid_file_set))
    for i in range(len(valid_file_set) // 2):
        dir_name = "DLC" + str(start_index)
        if os.path.exists(os.path.join(root_path, dir_name)):
            left_direction_file_set.append(dir_name)
        else:
            print("{} not exist, please check original data".format(dir_name))
        start_index += cycle
    print("left:", left_direction_file_set)

    start_index -= cycle
    for i in range(len(valid_file_set) // 2):
        dir_name = "DRC" + str(start_index)
        if os.path.exists(os.path.join(root_path, dir_name)):
            right_direction_file_set.append(dir_name)
        else:
            print("{} not exist, please check original data".format(dir_name))
        start_index -= cycle
    print("right", right_direction_file_set)

    # 制作朝左运动的偏移量表
    while True:
        print("开始制作左向数据")
        res = calibration_images("./left.csv", root_path, left_direction_file_set, L_line, angle, "L")
        if not res:
            # 正常退出
            return None
        print("开始制作右向数据")
        res = calibration_images("./right.csv", root_path, right_direction_file_set, R_line, angle, "R")
        break


if __name__ == '__main__':
    make_template(TARGET_PATH, -2104, 8, ANGLE, 118, 117)