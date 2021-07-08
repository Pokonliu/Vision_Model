import os
import math
import pandas
import numpy as np
import cv2.cv2 as cv2
import collections
from multiprocessing import Process
from Common import utils
from Control import Serialization
from Common.const import const

predict_framework = "pytorch"
if predict_framework == "pytorch":
    import torch
    import torch.backends.cudnn as cudnn
    from Predict.ShuffleNetV2.network import ShuffleNetV2
elif predict_framework == 'tensorflow':
    import tensorflow as tf


class ModeProcess:
    def __init__(self, s2p_queue, d2p_queue, v2p_control_flag, v2p_params_str):
        self.mode_process = Process(target=self.predict_handle, args=(s2p_queue, d2p_queue, v2p_control_flag, v2p_params_str, ))

    # 封装进程启动API
    def open(self):
        if not self.mode_process.is_alive():
            self.mode_process.start()

    # 封装进程结束API
    def close(self):
        if self.mode_process.is_alive():
            self.mode_process.terminate()

    def predict_handle(self, s2p_queue, d2p_queue, v2p_control_flag, v2p_params_str):
        print("MODE 进程开始执行")
        serial_queue = s2p_queue
        image_queue = d2p_queue
        predict_flag = v2p_control_flag
        sequence_file_name = v2p_params_str

        # 制作偏移表参数
        get_in_row = None
        get_in_direction = None
        params_table = self.init_csv(predict_flag)

        shift_count = 1
        stitch_index = 0
        start_x = None
        valid_flag = False

        position_dict = {"last": 0, "cur": 0}
        epoch_image_list = [np.zeros((const.NEEDLE_GRID_HIGH, 1)) for _ in range(const.TOTAL_NEEDLE_GRID_COUNT)]

        # 初始化
        net_work = self.init_model(framework=predict_framework)
        serialization_process = Serialization.SerializationProcess()

        while True:
            if not image_queue.empty() and not serial_queue.empty():
                print("I queue:", image_queue.qsize())
                print("S queue:", serial_queue.qsize())
                image_data = image_queue.get()
                direction, row, col = serial_queue.get()

                # 空闲模式
                if predict_flag.value == const.IDLE_MODE:
                    # TODO:空闲模式删除队列,后期在C++中优化进程间消息传输流程
                    print("get into idle")

                # 测试专用存图模式
                elif predict_flag.value == const.IO_WRITE_MODE:
                    utils.save_image_by_needle(image_data, direction, col, row)

                # 自动校准模式
                elif predict_flag.value == const.AUTO_CALIBRATION:
                    # TODO:此处逻辑待定，检测三光路是否需要循环三次
                    print("get into AUTO CALIBRATION")
                    success_count = 0
                    for i in range(const.PREDICT_DIRECTION_COUNT):
                        res, real_angle, real_midline = self.Auto_calibration(image_data)
                        print("成功检测出中线和旋转角度：", res, real_angle, real_midline)
                        if res:
                            success_count += 1
                            data = pandas.DataFrame({"position": "ALL", "angle": real_angle, "midline": real_midline},
                                                    columns=["position", "x_coordinate", "shift_count", "index", "angle", "midline"], index=[0])
                            data.to_csv(const.SHIFT_TABLE_FILES[i], mode="a", index=False, header=False)
                            params_table[i]["angle"] = real_angle
                            params_table[i]["midline"] = real_midline
                    if success_count == const.PREDICT_DIRECTION_COUNT:
                        predict_flag.value = const.MAKE_TABLE_MODE

                # 制作偏移表模式
                elif predict_flag.value == const.MAKE_TABLE_MODE:
                    print("get into make table", get_in_row, get_in_direction, row)
                    # 跳过非完整行
                    if not isinstance(get_in_row, int) and not isinstance(get_in_direction, int) and direction == "L":
                        get_in_row = row
                        get_in_direction = direction

                    if isinstance(get_in_row, int) and row - get_in_row == 3:
                        params_table = self.init_csv(predict_flag)
                        print("制作偏移表完成")

                    if isinstance(get_in_row, int) and 1 <= row - get_in_row < 3:
                        # TODO：读写CSV是否可以优化，每次写一行与缓存完所有数据一次写入的差异待测
                        for i in range(const.PREDICT_DIRECTION_COUNT):
                            # 参数初始化
                            continue_flag = False
                            x_coordinate = []
                            x_k_means = []

                            rotate_img = utils.rotate_image(image_data, params_table[i]["angle"])
                            window_img = rotate_img[params_table[i]["midline"] - const.WINDOW_X0_OFFSET:
                                                    params_table[i]["midline"] - const.WINDOW_X1_OFFSET,
                                                    const.WINDOW_Y0:
                                                    const.WINDOW_Y1]

                            # TODO:OpenCV超参的魔鬼数字后期移入const中
                            # 二值化
                            _, window_img = cv2.threshold(window_img, thresh=110, maxval=255, type=cv2.THRESH_BINARY)
                            # 边缘检测
                            edge_img = cv2.Canny(window_img, threshold1=110, threshold2=200)
                            # 霍夫直线检测
                            lines = cv2.HoughLinesP(edge_img, rho=1, theta=np.pi / 180, threshold=10, lines=0, minLineLength=10, maxLineGap=10)

                            if lines is not None:
                                parallel_lines_count = 0
                                for line in lines:
                                    if abs(line[0][0] - line[0][2]) > 7:
                                        parallel_lines_count += 1
                                    else:
                                        x_coordinate.append((line[0][0] + line[0][2]) / 2)
                                if parallel_lines_count > 1:
                                    continue_flag = True
                                x_coordinate.sort()
                                for x in x_coordinate:
                                    if len(x_k_means) == 0:
                                        x_k_means.append([x, [x]])
                                    else:
                                        for index in range(len(x_k_means)):
                                            if x_k_means[index][0] + 5 > x > x_k_means[index][0] - 5:
                                                x_k_means[index][1].append(x)
                                                x_k_means[index][0] = sum(x_k_means[index][1]) / len(x_k_means[index][1])
                                                break
                                        else:
                                            x_k_means.append([x, [x]])
                            true_center = self.center_filter([x_c[0] for x_c in x_k_means])

                            if 10 >= len(x_k_means) >= 2 and 5 >= len(true_center) >= 2 and not continue_flag:
                                valid_flag = True
                                if isinstance(start_x, float):
                                    start_x = self.search(true_center, start_x - const.WINDOW_Y0) + const.WINDOW_Y0
                                else:
                                    start_x = true_center[0] + const.WINDOW_Y0 if direction == "R" else true_center[-1] + const.WINDOW_Y0

                                if direction == "R":
                                    # 预算移动是否会超出窗口
                                    if start_x + shift_count * const.NEEDLE_GRID_WIDTH - const.RELATIVE_OFFSET > const.WINDOW_Y1 - 32:
                                        shift_count = 1
                                    elif const.WINDOW_Y0 + 32 > start_x + shift_count * const.NEEDLE_GRID_WIDTH - const.RELATIVE_OFFSET:
                                        shift_count = 2

                                elif direction == "L":
                                    # 预算移动是否会超出窗口
                                    if start_x - shift_count * const.NEEDLE_GRID_WIDTH + const.RELATIVE_OFFSET > const.WINDOW_Y1 - 32:
                                        shift_count = 2
                                    elif const.WINDOW_Y0 + 32 > start_x - shift_count * const.NEEDLE_GRID_WIDTH + const.RELATIVE_OFFSET:
                                        shift_count = 1

                                data = pandas.DataFrame({"position": "D%sC%d" % (direction, col), "x_coordinate": start_x, "shift_count": shift_count, "stitch_index": stitch_index},
                                                        columns=["position", "x_coordinate", "shift_count", "stitch_index", "angle", "midline"], index=[0])
                                data.to_csv(const.SHIFT_TABLE_FILES[i], mode="a", index=False, header=False)
                                # 修改下一次起始位置
                                start_x = start_x + (1 if direction == "R" else -1) * (shift_count * const.NEEDLE_GRID_WIDTH - const.RELATIVE_OFFSET)
                                # 更新下一次索引
                                stitch_index += (1 if direction == "R" else -1) * shift_count
                            else:
                                data = pandas.DataFrame({"position": "D%sC%d" % (direction, col), "x_coordinate": 'invalid', "shift_count": "invalid", "stitch_index": "invalid"},
                                                        columns=["position", "x_coordinate", "shift_count", "stitch_index", "angle", "midline"], index=[0])
                                data.to_csv(const.SHIFT_TABLE_FILES[i], mode="a", index=False, header=False)
                                # TODO:修改结尾处的数据
                                if valid_flag:
                                    stitch_index -= shift_count
                                    valid_flag = False
                                start_x = None

                # 制作模板模式
                elif predict_flag.value == const.MAKE_TEMPLATE_MODE:
                    # 逻辑与预测模式一样，只是增加了最后写模板的步骤
                    if not isinstance(get_in_row, int):
                        get_in_row = row
                    elif row != get_in_row:
                        print(position_dict["last"], position_dict["cur"])
                        if position_dict["last"] != position_dict["cur"]:
                            cv2.imwrite("./TROW-{}.jpg".format(get_in_row), np.hstack(epoch_image_list[min(position_dict["cur"], position_dict["last"]): max(position_dict["cur"], position_dict["last"])]))
                        position_dict["last"] = position_dict["cur"]
                        get_in_row = row
                    if direction in ["R", "L"]:
                        self.split_image(image_data, direction, col, params_table, epoch_image_list, position_dict)

                # 预测模式
                elif predict_flag.value == const.PREDICT_MODE:
                    # 逻辑与制作模板一样，只是增加了最后的比对的步骤
                    # TODO:需要按照模板来设置数据格式,后期C++固化数据，通过掩码来获取相应的结果
                    pass
                    # self.split_image(image_data, direction, col, csv_files, epoch_image_list, position_dict)

    @staticmethod
    def init_model(framework):
        if framework == "tensorflow":
            model = tf.keras.models.load_model(filepath=const.TF_MODEL_PATH)
        elif framework == "pytorch":
            cudnn.benchmark = True
            model = ShuffleNetV2(model_size="1.5x")
            checkpoint = torch.load(const.PYTORCH_MODEL_PATH)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            model.load_state_dict(checkpoint)

            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
        return model

    @staticmethod
    def init_csv(predict_flag):
        res = []
        for i in range(const.PREDICT_DIRECTION_COUNT):
            angle = 0
            midline = 0
            if not os.path.exists(const.SHIFT_TABLE_FILES[i]):
                data = pandas.DataFrame(columns=["position", "x_coordinate", "shift_count", "stitch_index", "angle", "midline"])
                data.to_csv(const.SHIFT_TABLE_FILES[i], mode="a", index=False, header=True)
                csv_file = pandas.read_csv(const.SHIFT_TABLE_FILES[i])
                # TODO: 逻辑待优化，模式需要外部触发
                predict_flag.value = const.AUTO_CALIBRATION
            else:
                csv_file = pandas.read_csv(const.SHIFT_TABLE_FILES[i])
                row_index = csv_file[csv_file.position == "ALL"].index.tolist()
                angle = float(csv_file.loc[row_index, "angle"].values)
                midline = round(float(csv_file.loc[row_index, "midline"].values))
                predict_flag.value = const.IDLE_MODE
            res.append({"csv_file": csv_file, "angle": angle, "midline": midline})
        return res

    @staticmethod
    def search(group, target):
        if target <= group[0]:
            return group[0]
        elif target >= group[-1]:
            return group[-1]

        left = 0
        right = len(group)
        while left < right:
            mid = left + ((right - left + 1) >> 1)
            if group[mid] > target:
                right = mid - 1
            else:
                left = mid
        return group[left] if abs(group[left] - target) < abs(group[left + 1] - target) else group[left + 1]

    @staticmethod
    def center_filter(center_set):
        true_center = []
        queue = collections.deque()
        queue.append([0, 1])
        shadow_queue = [[0, 1]]

        while queue:
            cur_p, next_p = queue.popleft()
            if next_p >= len(center_set):
                continue
            if 37 >= center_set[next_p] - center_set[cur_p] >= 26:
                true_center.append(center_set[cur_p])
                if next_p == len(center_set) - 1:
                    true_center.append(center_set[next_p])
            else:
                if [cur_p, next_p + 1] not in shadow_queue:
                    queue.append([cur_p, next_p + 1])
                    shadow_queue.append([cur_p, next_p + 1])
            if [next_p, next_p + 1] not in shadow_queue:
                queue.append([next_p, next_p + 1])
                shadow_queue.append([next_p, next_p + 1])
        return true_center

    @staticmethod
    def predict(framework, model, images):
        if framework == "tensorflow":
            res = model.predict(images)
        else:
            res = model(images)
        return res

    @staticmethod
    def split_image(image, direction, col, params_table, container, position):
        for i in range(const.PREDICT_DIRECTION_COUNT):
            rotate_image = utils.rotate_image(image, params_table[i]["angle"])

            row_index = params_table[i]["csv_file"][params_table[i]["csv_file"].position == "D{}C{}".format(direction, col)].index.tolist()
            if len(row_index) == 0:
                print("当前位置不存在于偏移表中,D{}C{}".format(direction, col))
                break

            x_coordinate = params_table[i]["csv_file"].loc[row_index, "x_coordinate"].tolist()[0]
            if x_coordinate == "invalid":
                break
            else:
                x_coordinate = round(float(x_coordinate))
            shift_count = int(params_table[i]["csv_file"].loc[row_index, "shift_count"].values)
            stitch_index = int(params_table[i]["csv_file"].loc[row_index, "stitch_index"].values)

            # 记录每次截取的区域
            position["cur"] = stitch_index + (1 if direction == "R" else -1) * shift_count

            # TODO: 后期需要截取上下针排，目前只截取了下针排
            y_coordinate = params_table[i]["midline"] - const.NEEDLE_GRID_HIGH

            for j in range(shift_count):
                if direction == "R":
                    cropped_region = rotate_image[y_coordinate:
                                                  y_coordinate + const.NEEDLE_GRID_HIGH,
                                                  x_coordinate + const.NEEDLE_GRID_WIDTH * j:
                                                  x_coordinate + const.NEEDLE_GRID_WIDTH * (j + 1)]
                    container[stitch_index + j] = cropped_region
                elif direction == "L":
                    cropped_region = rotate_image[y_coordinate:
                                                  y_coordinate + const.NEEDLE_GRID_HIGH,
                                                  x_coordinate - const.NEEDLE_GRID_WIDTH * (j + 1):
                                                  x_coordinate - const.NEEDLE_GRID_WIDTH * j]
                    container[stitch_index - 1 - j] = cropped_region

    @staticmethod
    def Auto_calibration(src):
        real_degree = 0
        real_midline = 0

        # 伽马变换
        gamma = np.uint8(np.power(src / 255.0, 4) * 255.0)
        # 直方图均衡化
        equal = cv2.equalizeHist(gamma)
        # 边缘检测
        edge = cv2.Canny(equal, threshold1=100, threshold2=200)
        # 霍夫线检测
        lines = cv2.HoughLinesP(edge, rho=1, theta=np.pi / 180, threshold=50, lines=0, minLineLength=10, maxLineGap=20)
        # 斜率检测
        angle_list = []
        if lines is not None:
            for line in lines:
                k = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
                degree = math.degrees(math.atan(k))
                if 30 < degree < 45:
                    angle_list.append(degree)
        else:
            print("1.This image couldn't find real midline,reason:HoughLinesP detect no line")
            return False, real_degree, real_midline
        if len(angle_list) == 0:
            print("2.This image couldn't find real midline,reason:angle detection zero")
            return False, real_degree, real_midline
        real_degree = sum(angle_list) / len(angle_list) + 90

        # 旋转图片
        rotate = utils.rotate_image(edge, real_degree)
        # 对旋转后的图片继续霍夫直线检测
        lines = cv2.HoughLinesP(rotate, rho=1, theta=np.pi / 180, threshold=50, lines=0, minLineLength=10, maxLineGap=20)
        # 直线域叠加
        cluster_list = []
        if lines is not None:
            for line in lines:
                # 过滤所有的垂线
                if abs(line[0][1] - line[0][3]) <= 3 and abs(line[0][0] - line[0][2]) >= 5:
                    continue
                # swap
                if line[0][1] > line[0][3]:
                    line[0][1], line[0][3] = line[0][3], line[0][1]

                if len(cluster_list) == 0:
                    cluster_list.append([line[0][1], line[0][3]])
                    continue
                for cluster in cluster_list:
                    if cluster[0] <= line[0][1] <= cluster[1] or cluster[0] <= line[0][3] <= cluster[1] or line[0][1] <= cluster[0] <= line[0][3] or line[0][1] <= cluster[1] <= line[0][3]:
                        cluster[0] = min(cluster[0], line[0][1], line[0][3])
                        cluster[1] = max(cluster[1], line[0][1], line[0][3])
                        break
                else:
                    cluster_list.append([line[0][1], line[0][3]])
        else:
            print("3.This image couldn't find real midline,reason:HoughLinesP detect no line")
            return False, real_degree, real_midline

        if len(cluster_list) != 2:
            print("4.This image couldn't find real midline,reason:cluster lens error {}".format(cluster_list))
            return False, real_degree, real_midline
        else:
            rotate_src = utils.rotate_image(src, real_degree)
            w, h = rotate_src.shape[:2]

            y_coordinate = [y for c in cluster_list for y in c]
            y_coordinate.sort()
            ROI_top = y_coordinate[1] - 5
            ROI_down = y_coordinate[2] + 5

            # 获取ROI
            roi = rotate_src[ROI_top: ROI_down, 0: h]
            # 二值化
            _, threshold = cv2.threshold(roi, thresh=25, maxval=255, type=cv2.THRESH_BINARY)
            # 霍夫直线检测
            lines = cv2.HoughLinesP(threshold, rho=1, theta=np.pi / 180, threshold=50, lines=0, minLineLength=10, maxLineGap=40)
            if lines is not None:
                line_y_cor = [(line[0][1] + line[0][3]) / 2 for line in lines]
                line_y_cor.sort()
            else:
                print("5.This image couldn't find real midline,reason:HoughLinesP detect no line")
                return False, real_degree, real_midline

            for index in range(len(line_y_cor) - 1):
                if 60 > line_y_cor[index + 1] - line_y_cor[index] > 40:
                    real_ROI_top = line_y_cor[index]
                    real_ROI_down = line_y_cor[index + 1]
                    break
            else:
                print("6.This image couldn't find real midline,line gap error, {}".format(line_y_cor))
                return False, real_degree, real_midline

            real_midline = (real_ROI_top + real_ROI_down) / 2 + y_coordinate[1] - 5
            return True, real_degree, real_midline
