import numpy as np
import tensorflow as tf
import utils
import Serialization
from const import const


def predict_process(predictQueue, predictProcessFlag, sequenceFileName):
    print("Predict 进程开始执行")
    predict_queue = predictQueue
    serialization_process = Serialization.SerializationProcess()
    net_work = tf.keras.models.load_model(filepath=const.TF_MODEL_PATH)
    predict_flag = predictProcessFlag
    sequence_file_name = sequenceFileName
    save_sequence_file_flag = False
    absolute_locations = init_absolute_coordinate()
    epoch_image_list = []
    while True:
        try:
            while predict_flag.value or not predict_queue.empty():
                save_sequence_file_flag = True
                if not predict_queue.empty():
                    print("queue:", predict_queue.qsize())
                    image_data, direction, row, col = predict_queue.get()
                    cur_image_index = 0
                    if direction == "R":
                        cur_image_index = (col - const.START_IMAGE_NUMBER_R) // 8
                    elif direction == "L":
                        cur_image_index = (col - const.START_IMAGE_NUMBER_L) // 8 + const.EACH_LINE_COUNTS
                    # TODO: 对于掉头出现的TL与TR如何处理(后期处理)

                    for i in range(const.PREDICT_DIRECTION_COUNT):
                        rotate_image = utils.rotate_image(image_data, const.PREDICT_ANGELS[i])
                        scale_image = utils.scale_image(rotate_image, const.PREDICT_SCALE[i])
                        for j in range(const.CROP_COUNT):
                            cropped_region = scale_image[absolute_locations[i][cur_image_index][1]:
                                                         absolute_locations[i][cur_image_index][1] + const.NEEDLE_GRID_HIGH,
                                                         absolute_locations[i][cur_image_index][0] + const.NEEDLE_GRID_WIDTH * j:
                                                         absolute_locations[i][cur_image_index][0] + const.NEEDLE_GRID_WIDTH * (j + 1)]
                            epoch_image_list.append(cropped_region)

                if len(epoch_image_list) > 500:
                    images = np.array(epoch_image_list)
                    predict_output = net_work.predict(images)
                    predict_category = np.argmax(predict_output, axis=1)
                    serialization_process.adding_sequence("".join([str(x) for x in predict_category.tolist()]))
                    epoch_image_list = []
            if save_sequence_file_flag and predict_queue.qsize() == 0:
                serialization_process.save_file(sequence_file_name.value)
                save_sequence_file_flag = False
        except Exception as error:
            print("Predict error occurred {}".format(error))


def calculate_absolute_coordinate(cur_num, cur_direction, cur_absolute_location_x, cur_absolute_location_y, relative_coordinates):
    # 首先要知道第一张图的序号，最左端图像的序号number_0
    # eg:
    start_num = const.START_IMAGE_NUMBER_R if cur_direction == "R" else const.START_IMAGE_NUMBER_L
    # 绝对坐标列表
    absolute_locations = []
    # 每个1/4个周期记为1次，8个1/4周期记一次数,计算每个序号下图像对应的索引
    img_index = (cur_num - start_num) // 8 + (0 if cur_direction == "R" else const.EACH_LINE_COUNTS)
    # 读取坐标模板中对应的x方向的相对坐标
    relative_x_m = relative_coordinates[img_index]
    # 得到坐标模板的总长度，遍历每一个索引，求出绝对坐标
    for i in range(len(relative_coordinates)):
        # 如果索引中图像相对0的坐标值比传入图像相对0的坐标值小，表示向左移动，x值变小，即减去
        absolute_location_i = cur_absolute_location_x - (relative_x_m - int(relative_coordinates[i]))
        absolute_locations.append([absolute_location_i, cur_absolute_location_y])
    # 返回绝对坐标列表
    return absolute_locations


def init_absolute_coordinate():
    # TODO：初始化三组绝对坐标
    result = []
    for i in range(const.PREDICT_DIRECTION_COUNT):
        # TODO：第一次读取图片，从串口获取对应图片序号，通过图像预处理方式来确定三个取像区域的绝对坐标
        img_index = const.START_IMAGE_INDEX
        img_direction = const.START_IMAGE_DIRECTION
        start_coordinate_x = const.START_IMAGE_COORDINATE_X
        start_coordinate_y = const.START_IMAGE_COORDINATE_Y
        result.append(calculate_absolute_coordinate(img_index, img_direction, start_coordinate_x, start_coordinate_y, utils.read_csv(const.COORDINATE_FILES[i])))
    return result
