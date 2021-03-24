import os
import utils
import cv2.cv2 as cv2
from const import const


# TODO：IO Process后期在采集速率与预测速率达到一致后废弃
def io_process(ioQueue, predictQueue, ioProcessFlag, imageDefaultPath):
    print("I/O 进程开始执行")
    if not os.path.exists(imageDefaultPath.value):
        os.mkdir(imageDefaultPath.value)
    io_queue = ioQueue
    predict_queue = predictQueue
    io_process_flag = ioProcessFlag
    frame_count = 0
    while True:
        while io_process_flag.value or not io_queue.empty():
            if not io_queue.empty():
                # 正常逻辑
                # image_data = io_queue.get()
                # for i in range(const.CROP_COUNT):
                #     cropped_region = image_data[const.STARTING_COORDINATES[1]: const.STARTING_COORDINATES[1] + 100,
                #                                 const.STARTING_COORDINATES[0] + const.NEEDLE_GRID_WIDTH * i: const.STARTING_COORDINATES[0] + const.NEEDLE_GRID_WIDTH * (i + 1)]
                #     file_name = "frame_%d_cropped_%d.jpg" % (frame_count, i)
                #     print('存入图片{}'.format(file_name))
                #     predict_queue.put(file_name)
                #     cv2.imwrite(os.path.join("./temp/images", file_name), cropped_region)

                # 测试逻辑
                print("queue:", io_queue.qsize())
                image_data, direction, row, col = io_queue.get()
                utils.save_image_by_needle(image_data, direction, col, row)
                # frame_count += 1
