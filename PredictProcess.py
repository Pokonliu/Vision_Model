import os
import numpy as np
import cv2.cv2 as cv2
import cv2.dnn as dnn
import tensorflow as tf
import utils
import Serialization
from const import const


# opencv版本
# def predict_process(predictQueue, predictProcessFlag, sequenceFileName):
#     print("Predict 进程开始执行")
#     predict_queue = predictQueue
#     serialization_process = Serialization.SerializationProcess()
#     net_work = dnn.readNetFromTensorflow(model=const.MODEL_PATH)
#     predict_flag = predictProcessFlag
#     sequence_file_name = sequenceFileName
#     save_sequence_file_flag = True
#     while True:
#         try:
#             while predict_flag.value or not predict_queue.empty():
#                 save_sequence_file_flag = True
#                 if not predict_queue.empty():
#                     file_name = predict_queue.get()
#                     print(predict_queue.qsize(), " ", file_name)
#                     target_image_path = os.path.join(r".\temp\images", file_name)
#                     src = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
#                     print("文件路径={}".format(target_image_path))
#                     print("文件形状={}".format(src.shape))
#                     net_work.setInput(dnn.blobFromImage(src, size=(118, 64), swapRB=True, crop=False))
#                     cv_out = net_work.forward()
#                     category = np.argmax(cv_out)
#                     print("{}预测结果为{}".format(file_name, category))
#                     serialization_process.adding_sequence(str(category))
#                     # os.remove(target_image_path)
#             if save_sequence_file_flag:
#                 serialization_process.save_file(sequence_file_name.value)
#                 save_sequence_file_flag = False
#         except Exception as error:
#             print("Predict error occurred {}".format(error))


# Tensorflow版本
def predict_process(predictQueue, predictProcessFlag, sequenceFileName):
    print("Predict 进程开始执行")
    predict_queue = predictQueue
    serialization_process = Serialization.SerializationProcess()
    net_work = tf.keras.models.load_model(filepath=const.TF_MODEL_PATH)
    predict_flag = predictProcessFlag
    sequence_file_name = sequenceFileName
    save_sequence_file_flag = True
    images = []
    while True:
        # TODO：初始化三组绝对坐标
        for i in range(const.PREDICT_DIRECTION_COUNT):
            # TODO：第一次读取图片，从串口获取对应图片序号，通过图像预处理方式来确定三个取像区域的绝对坐标
            img_index = 1
            image_coordinate = 1
            start_coordinate_x, start_coordinate_y = calculate_absolute_coordinate(img_index, image_coordinate, utils.read_csv(const.COORDINATE_FILES[i]))
        try:
            while predict_flag.value or not predict_queue.empty():
                save_sequence_file_flag = True
                if not predict_queue.empty():
                    print("queue:", predict_queue.qsize())
                    image_data, direction, row, col = predict_queue.get()
                    cur_image_index = (col - const.START_IMAGE_NUMBER) // 8

                    for i in range(const.PREDICT_DIRECTION_COUNT):
                        rotate_image = utils.rotate_image(image_data, const.PREDICT_ANGELS[i])
                        for j in range(const.CROP_COUNT):
                            cropped_region = rotate_image[const.STARTING_COORDINATES[1]: const.STARTING_COORDINATES[1] + const.NEEDLE_GRID_HIGH,
                                                          const.STARTING_COORDINATES[0] + const.NEEDLE_GRID_WIDTH * j: const.STARTING_COORDINATES[0] + const.NEEDLE_GRID_WIDTH * (j + 1)]
                            images.append(cropped_region)

                if len(images) > 500:
                    images = np.array(images)
                    cv_out = net_work.predict(images)
                    print(cv_out)
                    print(cv_out.shape)
                    images = []
            if save_sequence_file_flag:
                serialization_process.save_file(sequence_file_name.value)
                save_sequence_file_flag = False
        except Exception as error:
            print("Predict error occurred {}".format(error))


def calculate_absolute_coordinate(number_m, absolute_location_m, relative_coordinates):
    # 首先要知道第一张图的序号，最左端图像的序号number_0
    # eg:
    number_0 = -100
    # 绝对坐标列表
    absolute_locations = []
    # 每个1/4个周期记为1次，8个1/4周期记一次数,计算每个序号下图像对应的索引
    img_index = (number_m - number_0) / 8
    # 读取坐标模板中对应的x方向的相对坐标
    relative_x_m = relative_coordinates[img_index]
    # 得到坐标模板的总长度，遍历每一个索引，求出绝对坐标
    for i in range(len(relative_coordinates)):
        # 如果索引中图像相对0的坐标值比传入图像相对0的坐标值小，表示向左移动，x值变小，即减去
        absolute_location_i = absolute_location_m - (relative_x_m - int(relative_coordinates[i]))
        absolute_locations.append(absolute_location_i)
    # 返回绝对坐标列表
    return absolute_locations

# if __name__ == '__main__':
    # import tensorflow as tf
    # import numpy as np
    # import time
    #
    # net_work = dnn.readNetFromTensorflow(model=const.MODEL_PATH)
    # start = time.time()
    # for root, dirs, files in os.walk("./temp/images"):
    #     for file in files:
    #         target_image_path = os.path.join(root, file)
    #         src = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
    #         net_work.setInput(dnn.blobFromImage(src, size=(118, 64), swapRB=True, crop=False))
    #         cv_out = net_work.forward()
    #         category = np.argmax(cv_out)
    # end = time.time()
    # print("OpenCV predict single image each group spend:{}".format(end - start))
    #
    # start = time.time()
    # for root, dirs, files in os.walk("./temp/images"):
    #     images = []
    #     for file in files:
    #         target_image_path = os.path.join(root, file)
    #         images.append(cv2.imread(target_image_path, cv2.IMREAD_COLOR))
    #         if len(images) == 10:
    #             net_work.setInput(dnn.blobFromImages(images, size=(118, 64), swapRB=True, crop=False))
    #             cv_out = net_work.forward()
    #             # print(cv_out)
    #             images = []
    # end = time.time()
    # print("OpenCV predict 10 image each group spend:{}".format(end - start))
    #
    # start = time.time()
    # for root, dirs, files in os.walk("./temp/images"):
    #     images = []
    #     for file in files:
    #         target_image_path = os.path.join(root, file)
    #         images.append(cv2.imread(target_image_path, cv2.IMREAD_COLOR))
    #         if len(images) == 100:
    #             net_work.setInput(dnn.blobFromImages(images, size=(118, 64), swapRB=True, crop=False))
    #             cv_out = net_work.forward()
    #             # print(cv_out)
    #             images = []
    # end = time.time()
    # print("OpenCV predict 100 image each group spend:{}".format(end - start))

    # net_work = tf.keras.models.load_model("./stf_new_model")
    # start = time.time()
    # for root, dirs, files in os.walk("./temp/images"):
    #     for file in files:
    #         target_image_path = os.path.join(root, file)
    #         target = cv2.resize(src=cv2.imread(target_image_path, cv2.IMREAD_COLOR), dsize=(118, 64))
    #         ans = net_work.predict(target.reshape(1, 64, 118, 3))
    # end = time.time()
    # print("Tensorflow predict single image each group spend:{}".format(end - start))
    #
    # start = time.time()
    # for root, dirs, files in os.walk("./temp/images"):
    #     images = []
    #     for file in files:
    #         target_image_path = os.path.join(root, file)
    #         images.append(cv2.resize(src=cv2.imread(target_image_path, cv2.IMREAD_COLOR), dsize=(118, 64)))
    #         if len(images) == 10:
    #             images = np.array(images)
    #             ans = net_work.predict(images)
    #             images = []
    # end = time.time()
    # print("Tensorflow predict 10 image each group spend:{}".format(end - start))
    #
    # start = time.time()
    # for root, dirs, files in os.walk("./temp/images"):
    #     images = []
    #     for file in files:
    #         target_image_path = os.path.join(root, file)
    #         images.append(cv2.resize(src=cv2.imread(target_image_path, cv2.IMREAD_COLOR), dsize=(118, 64)))
    #         if len(images) == 100:
    #             images = np.array(images)
    #             ans = net_work.predict(images)
    #             images = []
    # end = time.time()
    # print("Tensorflow predict 100 image each group spend:{}".format(end - start))
    #
    # start = time.time()
    # for root, dirs, files in os.walk("./temp/images"):
    #     images = []
    #     for file in files:
    #         target_image_path = os.path.join(root, file)
    #         images.append(cv2.resize(src=cv2.imread(target_image_path, cv2.IMREAD_COLOR), dsize=(118, 64)))
    #         if len(images) == 500:
    #             images = np.array(images)
    #             ans = net_work.predict(images)
    #             images = []
    # end = time.time()
    # print("Tensorflow predict 500 image each group spend:{}".format(end - start))
