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
    epoch_image_list = []
    while True:
        # TODO：初始化三组绝对坐标
        absolute_locations = []
        for i in range(const.PREDICT_DIRECTION_COUNT):
            # TODO：第一次读取图片，从串口获取对应图片序号，通过图像预处理方式来确定三个取像区域的绝对坐标
            img_index = 1
            start_coordinate_x = 1
            start_coordinate_y = 1
            absolute_locations.append(calculate_absolute_coordinate(img_index, start_coordinate_x, start_coordinate_y, utils.read_csv(const.COORDINATE_FILES[i])))
        try:
            while predict_flag.value or not predict_queue.empty():
                save_sequence_file_flag = True
                if not predict_queue.empty():
                    print("queue:", predict_queue.qsize())
                    image_data, direction, row, col = predict_queue.get()
                    cur_image_index = (col - const.START_IMAGE_NUMBER) // 8

                    for i in range(const.PREDICT_DIRECTION_COUNT):
                        rotate_image = utils.rotate_image(image_data, const.PREDICT_ANGELS[i])
                        scale_image = utils.scale_image(rotate_image, const.PREDICT_SCALE[i])
                        for j in range(const.CROP_COUNT):
                            cropped_region = scale_image[absolute_locations[i][1][cur_image_index]:
                                                         absolute_locations[i][1][cur_image_index] + const.NEEDLE_GRID_HIGH,
                                                         absolute_locations[i][0][cur_image_index] + const.NEEDLE_GRID_WIDTH * j:
                                                         absolute_locations[i][0][cur_image_index] + const.NEEDLE_GRID_WIDTH * (j + 1)]
                            epoch_image_list.append(cropped_region)

                if len(epoch_image_list) > 500:
                    images = np.array(epoch_image_list)
                    predict_output = net_work.predict(images)
                    predict_category = np.argmax(predict_output, axis=1)
                    serialization_process.adding_sequence("".join([str(x) for x in predict_category.tolist()]))
                    epoch_image_list = []
            if save_sequence_file_flag:
                serialization_process.save_file(sequence_file_name.value)
                save_sequence_file_flag = False
        except Exception as error:
            print("Predict error occurred {}".format(error))


def calculate_absolute_coordinate(number_m, absolute_location_m_x, absolute_location_m_y, relative_coordinates):
    # 首先要知道第一张图的序号，最左端图像的序号number_0
    # eg:
    number_0 = const.START_IMAGE_NUMBER
    # 绝对坐标列表
    absolute_locations = []
    # 每个1/4个周期记为1次，8个1/4周期记一次数,计算每个序号下图像对应的索引
    img_index = (number_m - number_0) // 8
    # 读取坐标模板中对应的x方向的相对坐标
    relative_x_m = relative_coordinates[img_index]
    # 得到坐标模板的总长度，遍历每一个索引，求出绝对坐标
    for i in range(len(relative_coordinates)):
        # 如果索引中图像相对0的坐标值比传入图像相对0的坐标值小，表示向左移动，x值变小，即减去
        absolute_location_i = absolute_location_m_x - (relative_x_m - int(relative_coordinates[i]))
        absolute_locations.append([absolute_location_i, absolute_location_m_y])
    # 返回绝对坐标列表
    return absolute_locations


# if __name__ == '__main__':
#     import tensorflow as tf
#     import numpy as np
#     import time
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
    #         target = cv2.resize(src=cv2.imread(target_image_path, cv2.IMREAD_COLOR), dsize=(39, 89))
    #         ans = net_work.predict(target.reshape(1, 89, 39, 3))
    #         out = np.argmax(ans, axis=1)
    #         print(file, out)
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
    #         images.append(cv2.resize(src=cv2.imread(target_image_path, cv2.IMREAD_COLOR), dsize=(39, 89)))
    #         if len(images) == 500:
    #             images = np.array(images)
    #             ans = net_work.predict(images)
    #             # print(ans)
    #             print(ans.shape)
    #             print(type(ans))
    #             out = np.argmax(ans, axis=1)
    #             print(out)
    #             print(type(out))
    #             print(out.shape)
    #             # out = out.tolist()
    #             # print(out)
    #             # print(type(out))
    #             # print(list(map(lambda x: str(x), out)))
    #             print("".join([str(x) for x in out.tolist()]))
    #             images = []
    # end = time.time()
    # print("Tensorflow predict 500 image each group spend:{}".format(end - start))
