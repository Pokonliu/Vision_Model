import os
import numpy as np
from Common import utils
from Control import Serialization
from Common.const import const

# predict_framework = "pytorch"
# if predict_framework == "pytorch":
#     import torch
#     import torch.backends.cudnn as cudnn
#     import torchvision
#     from Predict.network import ShuffleNetV2
# elif predict_framework == 'tensorflow':
#     import tensorflow as tf


# TODO：兼容TF、Pytorch、TensorRT三种预测模式
def PredictProcess(predictQueue, serialQueue, predictProcessFlag, sequenceFileName):
    print("Predict 进程开始执行")
    predict_queue = predictQueue
    serial_queue = serialQueue
    serialization_process = Serialization.SerializationProcess()
    net_work = init_model(framework=predict_framework)
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
                    image_data = predict_queue.get()
                    direction, row, col = serial_queue.get()
                    cur_image_index = 0
                    if direction == "R":
                        cur_image_index = (col - const.START_IMAGE_NUMBER_R) // 8
                    elif direction == "L":
                        cur_image_index = (col - const.START_IMAGE_NUMBER_L) // 8 + const.EACH_LINE_COUNTS
                    # TODO: 对于掉头出现的TL与TR如何处理(后期处理)

                    for i in range(const.PREDICT_DIRECTION_COUNT):
                        rotate_image = utils.rotate_image(image_data, const.PREDICT_ANGELS[i])
                        for j in range(const.CROP_COUNT):
                            cropped_region = rotate_image[absolute_locations[i][cur_image_index][1]:
                                                          absolute_locations[i][cur_image_index][1] + const.NEEDLE_GRID_HIGH,
                                                          absolute_locations[i][cur_image_index][0] + const.NEEDLE_GRID_WIDTH * j:
                                                          absolute_locations[i][cur_image_index][0] + const.NEEDLE_GRID_WIDTH * (j + 1)]
                            epoch_image_list.append(cropped_region)

                if len(epoch_image_list) > 500:
                    images = np.array(epoch_image_list)
                    predict_output = predict(framework=predict_framework, model=net_work, images=images)
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


def init_model(framework):
    if framework == "tensorflow":
        model = tf.keras.models.load_model(filepath=const.TF_MODEL_PATH)
    elif framework == "pytorch":
        cudnn.benchmark = True
        model = ShuffleNetV2(model_size="1.5x")
        checkpoint = torch.load(const.PYTORCH_MODEL_PATH)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint.items()})  # strip the names

        if torch.cuda.is_available():
            model = model.cuda()

        model.eval()
    return model


def predict(framework, model, images):
    if framework == "tensorflow":
        res = model.predict(images)
    else:
        res = model(images)
    return res


if __name__ == '__main__':
    import time
    import cv2.cv2 as cv2
    m = init_model(predict_framework)
    tran = torchvision.transforms.ToTensor()
    start = time.time()
    for root, dirs, files in os.walk(r"C:\Users\AR-LAB\Desktop\data and code\data\original_stitch\train\1"):
        images = []
        for file in files:
            target_image_path = os.path.join(root, file)
            images.append(tran(cv2.imread(target_image_path, cv2.IMREAD_COLOR)))
            # img = tran(cv2.imread(target_image_path, cv2.IMREAD_COLOR))
            # img = torch.unsqueeze(img, 0)
            # img = img.cuda()
            # res = predict(predict_framework, m, img)
            # print(res)

            if len(images) == 500:
                images = torch.tensor([item.cpu().detach().numpy() for item in images]).cuda()
                # images = np.array(images)
                # images = tran(images)
                res = predict(predict_framework, m, images)
                print(res.shape)
                # predict_category = np.argmax(res, axis=1)
                # print(predict_category)
                images = []
    end = time.time()
    print("Pytorch predict 1 image each group spend:{}".format(end - start))

