import os
import numpy as np
import cv2.cv2 as cv2
import cv2.dnn as dnn
import Serialization
from const import const


def predict_process(predictQueue, predictProcessFlag, sequenceFileName):
    print("Predict 进程开始执行")
    predict_queue = predictQueue
    serialization_process = Serialization.SerializationProcess()
    net_work = dnn.readNetFromTensorflow(model=const.MODEL_PATH)
    predict_flag = predictProcessFlag
    sequence_file_name = sequenceFileName
    save_sequence_file_flag = True
    while True:
        try:
            while predict_flag.value or not predict_queue.empty():
                save_sequence_file_flag = True
                if not predict_queue.empty():
                    file_name = predict_queue.get()
                    print(predict_queue.qsize(), " ", file_name)
                    target_image_path = os.path.join(r".\temp\images", file_name)
                    src = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
                    print("文件路径={}".format(target_image_path))
                    print("文件形状={}".format(src.shape))
                    net_work.setInput(dnn.blobFromImage(src, size=(118, 64), swapRB=True, crop=False))
                    cv_out = net_work.forward()
                    category = np.argmax(cv_out)
                    print("{}预测结果为{}".format(file_name, category))
                    serialization_process.adding_sequence(str(category))
                    # os.remove(target_image_path)
            if save_sequence_file_flag:
                serialization_process.save_file(sequence_file_name.value)
                save_sequence_file_flag = False
        except Exception as error:
            print("Predict error occurred {}".format(error))


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
