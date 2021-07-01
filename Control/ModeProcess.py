import os
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
        pre_row = -1
        make_table_flag = False
        loop_count = 0
        shift_count = 1
        stitch_index = 0
        start_x = None
        start_flag = False

        # 预测、制作模板参数
        epoch_image_list = [0 for _ in range(const.TOTAL_NEEDLE_GRID_COUNT)]

        # 初始化
        csv_files = self.init_csv()
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

                # 制作偏移表模式
                elif predict_flag.value == const.MAKE_TABLE_MODE:
                    print("get into make table")
                    if pre_row == -1:
                        pre_row = row
                    elif row != pre_row:
                        loop_count += 1
                        pre_row = row
                        make_table_flag = True

                    if loop_count == 3:
                        make_table_flag = False
                        csv_files = self.init_csv()
                        predict_flag.value = const.IDLE_MODE
                        print("制作偏移表完成")

                    if make_table_flag:
                        for i in range(const.PREDICT_DIRECTION_COUNT):
                            # 参数初始化
                            continue_flag = False
                            x_coordinate = []
                            x_k_means = []

                            rotate_img = utils.rotate_image(image_data, const.SPIN_ANGELS[i])
                            window_img = rotate_img[const.WINDOW_X0: const.WINDOW_X1, const.WINDOW_Y0: const.WINDOW_Y1]

                            # TODO:OpenCV超参的魔鬼数字后期移入const中
                            # 二值化
                            _, window_img = cv2.threshold(window_img, thresh=110, maxval=255, type=cv2.THRESH_BINARY)
                            # 边缘检测
                            edge_img = cv2.Canny(window_img, threshold1=100, threshold2=200)
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

                            # TODO: 进行二次检测，对一些异常center进行筛除
                            # TODO: 第一次检测出边缘就会进入该执行逻辑
                            # TODO: 第一次不允许在视野中出现梭子
                            if 10 >= len(x_k_means) >= 2 and 5 >= len(true_center) >= 2 and not continue_flag:
                                if start_flag:
                                    start_x = true_center[0] + const.WINDOW_Y0 if direction == "R" else true_center[-1] + const.WINDOW_Y0
                                else:
                                    start_x = self.search(true_center, start_x - const.WINDOW_Y0) + const.WINDOW_Y0

                                # 写入CSV
                                data = pandas.DataFrame({"position": "D%sC%d" % (direction, col), "x_coordinate": start_x, "shift_count": shift_count, "stitch_index": stitch_index},
                                                        columns=["position", "x_coordinate", "shift_count", "stitch_index"], index=[0])
                                data.to_csv(const.SHIFT_TABLE_FILES[i], mode="a", index=False, header=False)

                                if direction == "R":
                                    # 预算移动是否会超出窗口
                                    if start_x + shift_count * const.NEEDLE_GRID_WIDTH - const.RELATIVE_OFFSET > const.WINDOW_Y1 - 32:
                                        shift_count = 1
                                    elif const.WINDOW_Y0 + 32 > start_x + shift_count * const.NEEDLE_GRID_WIDTH - const.RELATIVE_OFFSET:
                                        shift_count = 2
                                    # 修改下一次起始位置
                                    start_x = start_x + shift_count * const.NEEDLE_GRID_WIDTH - const.RELATIVE_OFFSET

                                elif direction == "L":
                                    # 预算移动是否会超出窗口
                                    if start_x - shift_count * const.NEEDLE_GRID_WIDTH + const.RELATIVE_OFFSET > const.WINDOW_Y1 - 32:
                                        shift_count = 2
                                    elif const.WINDOW_Y0 + 32 > start_x - shift_count * const.NEEDLE_GRID_WIDTH + const.RELATIVE_OFFSET:
                                        shift_count = 1
                                    # 修改下一次起始位置
                                    start_x = start_x - shift_count * const.NEEDLE_GRID_WIDTH + const.RELATIVE_OFFSET

                                stitch_index += shift_count * (1 if direction == "R" else -1)
                                start_flag = False
                            else:
                                # TODO：非工作区域，或者计算出现异常可能会（需要实际测试）
                                data = pandas.DataFrame({"position": "D%sC%d" % (direction, col), "x_coordinate": 'invalid', "shift_count": "invalid", "stitch_index": "valid"},
                                                        columns=["position", "x_coordinate", "shift_count", "index"], index=[0])
                                data.to_csv(const.SHIFT_TABLE_FILES[i], mode="a", index=False, header=False)
                                start_flag = True

                # 制作模板模式
                elif predict_flag.value == const.MAKE_TEMPLATE_MODE:
                    # 逻辑与预测模式一样，只是增加了最后写模板的步骤
                    # TODO：目前测试行图像的切割是否精确
                    if pre_row == -1:
                        pre_row = row
                    elif row != pre_row:
                        cv2.imwrite("./TROW-{}.jpg".format(pre_row), np.hstack(epoch_image_list))
                        # TODO：epoch_image_list应该如何更新待定
                        epoch_image_list = [0 for _ in range(const.TOTAL_NEEDLE_GRID_COUNT)]
                        pre_row = row
                    if direction in ["R", "L"]:
                        self.split_image(image_data, direction, col, csv_files, epoch_image_list)

                # 预测模式
                elif predict_flag.value == const.PREDICT_MODE:
                    # 逻辑与制作模板一样，只是增加了最后的比对的步骤
                    # TODO:需要按照模板来设置数据格式,后期C++固化数据，通过掩码来获取相应的结果
                    if row != pre_row:
                        pass
                    self.split_image(image_data, direction, col, csv_files, epoch_image_list)

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
    def init_csv():
        csv_files = []
        for i in range(const.PREDICT_DIRECTION_COUNT):
            if not os.path.exists(const.SHIFT_TABLE_FILES[i]):
                data = pandas.DataFrame(columns=["position", "x_coordinate", "shift_count", "stitch_index"])
                data.to_csv(const.SHIFT_TABLE_FILES[i], mode="a", index=False, header=True)
            csv_files.append(pandas.read_csv(const.SHIFT_TABLE_FILES[i]))
        return csv_files

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
            if 34 >= center_set[next_p] - center_set[cur_p] >= 27:
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
    def split_image(image, direction, col, csv_files, container):
        for i in range(const.PREDICT_DIRECTION_COUNT):
            rotate_image = utils.rotate_image(image, const.SPIN_ANGELS[i])

            x_coordinate = csv_files[i][csv_files[i]["position"].isin(["D{}C{}".format(direction, col)])]["x_coordinate"].values
            shift_count = csv_files[i][csv_files[i]["position"].isin(["D{}C{}".format(direction, col)])]["shift_count"].values
            stitch_index = csv_files[i][csv_files[i]["position"].isin(["D{}C{}".format(direction, col)])]["stitch_index"].values
            # TODO: y轴坐标如何自动确定
            # TODO: 当角度发生变化如何修正
            y_coordinate = 300

            if x_coordinate == "invalid":
                continue

            x_coordinate = int(float(x_coordinate))
            for j in range(int(shift_count)):
                if direction == "R":
                    cropped_region = rotate_image[y_coordinate:
                                                  y_coordinate + const.NEEDLE_GRID_HIGH,
                                                  x_coordinate + const.NEEDLE_GRID_WIDTH * j:
                                                  x_coordinate + const.NEEDLE_GRID_WIDTH * (j + 1)]
                    container[int(stitch_index) + j] = cropped_region
                elif direction == "L":
                    cropped_region = rotate_image[y_coordinate:
                                                  y_coordinate + const.NEEDLE_GRID_HIGH,
                                                  x_coordinate - const.NEEDLE_GRID_WIDTH * (j + 1):
                                                  x_coordinate - const.NEEDLE_GRID_WIDTH * j]
                    container[int(stitch_index) - 1 - j] = cropped_region
