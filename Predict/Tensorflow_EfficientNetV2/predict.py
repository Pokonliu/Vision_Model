import os
import json
import glob
import time

import numpy as np

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from Predict.Tensorflow_EfficientNetV2.EfficientNetV2 import efficientnetv2_s as create_model


def main():
    num_classes = 3

    im_height = 90
    im_width = 32

    # load model
    model = create_model(num_classes=num_classes)
    model.build((1, 92, 32, 3))
    model.summary()
    weights_path = './save_weights/efficientnetv2.ckpt'

    assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)


    # load class
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # load image
    # start = time.time()
    # for root, dirs, files in os.walk(r"E:\NAR\Predict\test\1"):
    #     images = []
    #     for file in files:
    #         img_path = os.path.join(root, file)
    #         assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    #         img = Image.open(img_path)
    #         img = img.resize((im_width, im_height))
    #
    #         img = np.array(img).astype(np.float32)
    #         img = (img / 255. - 0.5) / 0.5
    #         images.append(img)
    #         if len(images) == 500:
    #             images = tf.stack(images, axis=0)
    #             res = model.predict(images)
    #             images = []
    # end = time.time()
    # print(end - start)
    # #
    # start = time.time()
    # for root, dirs, files in os.walk(r"E:\NAR\Predict\test\1"):
    #     images = []
    #     for file in files:
    #         img_path = os.path.join(root, file)
    #         assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    #         img = Image.open(img_path)
    #         img = img.resize((im_width, im_height))
    #
    #         img = np.array(img).astype(np.float32)
    #         img = (img / 255. - 0.5) / 0.5
    #         images.append(img)
    #         if len(images) == 500:
    #             images = np.array(images)
    #             res = model.predict(images)
    #             images = []
    # end = time.time()
    # print(end - start)
    #
    # start = time.time()
    # for root, dirs, files in os.walk(r"E:\NAR\Predict\test\1"):
    #     images = []
    #     for file in files:
    #         img_path = os.path.join(root, file)
    #         assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    #         img = Image.open(img_path)
    #         img = img.resize((im_width, im_height))
    #
    #         img = np.array(img).astype(np.float32)
    #         img = (img / 255. - 0.5) / 0.5
    #         images.append(img)
    #         if len(images) == 500:
    #             images = tf.stack(images, axis=0)
    #             res = model.predict(images)
    #             images = []
    # end = time.time()
    # print(end - start)

    start = time.time()
    predict_time = 0
    for root, dirs, files in os.walk(r"E:\NAR\Predict\test\0"):
        images = []
        for file in files:
            img_path = os.path.join(root, file)
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            img = img.resize((im_width, im_height))

            img = np.array(img).astype(np.float32)
            img = (img / 255. - 0.5) / 0.5
            images.append(img)
            if len(images) == 500:
                images = np.array(images)
                a = time.time()
                res = model.predict(images)
                print(tf.argmax(res, axis=1))
                b = time.time()
                predict_time += b - a
                images = []
    end = time.time()
    print("total time: ", end - start)
    print("predict time: ", predict_time)


if __name__ == '__main__':
    main()
