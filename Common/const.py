import os

import cv2.cv2 as cv2

from DataFlow.MvImport.MvCameraControl_class import *


class Const(object):
    class ConsError(TypeError):
        pass

    class ConstCaseError(ConsError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise (self.ConsError, "Can't change const.%s" % name)
        if not name.isupper():
            raise (self.ConstCaseError, "const name '%s' is not all uppercase" % name)
        self.__dict__[name] = value


const = Const()
# Video params
const.VIDEO_CONTROL_NONE = 0
const.VIDEO_SLOWER_CLICKED_FLAG = -1
const.VIDEO_FASTER_CLICKED_FLAG = 1
const.VIDEO_SLIDER_MOVED_FLAG = 2
const.VIDEO_FRAME_SINGLE_STEP = 200

# Predict process params
const.PREDICT_PROCESS_STARTING = 1
const.PREDICT_PROCESS_STOPPING = 0

# Predict mode
const.PREDICT_MAKE_TEMPLATE = 0
const.PREDICT_TRUE_SEQUENCE = 1

# IO process params
const.IO_PROCESS_STARTING = 1
const.IO_PROCESS_STOPPING = 0

# Image preprocessing params
const.TOP_SIDE_ANGLE = 13
const.SCALE_RATIO = 1
const.FILLING_COLOR = 0
const.CROP_COUNT = 4

const.START_IMAGE_NUMBER_R = -2014
const.START_IMAGE_NUMBER_L = 2128
const.EACH_LINE_COUNTS = 529

# Train params
const.PYTORCH_MODEL_PATH = "./Predict/ckpt/Shufflenet/model_best.pth"
const.TF_MODEL_PATH = "./Predict/Tensorflow_EfficientNetV2/save_weights/efficientnetv2.ckpt"

# TODO：预加载超参数，后期通过预处理寻找边缘
const.START_IMAGE_INDEX = -1288
const.START_IMAGE_DIRECTION = "L"
const.START_IMAGE_COORDINATE_X = 250
const.START_IMAGE_COORDINATE_Y = 150

# Image convert
const.GRAY2RGB = cv2.COLOR_GRAY2RGB
const.BGR2RGB = cv2.COLOR_BGR2RGB

# Data flow params
const.DATA_FLOW_STOP = 0
const.DATA_FLOW_IDLE = 1
const.DATA_FLOW_CHANGE = 2
const.DATA_FLOW_INIT = 3
const.DATA_FLOW_RUN = 4
const.DATA_FLOW_RELEASE = 5

# Source type
const.SOURCE_TYPE_NONE = 0
const.SOURCE_TYPE_IMAGE = 1
const.SOURCE_TYPE_VIDEO = 2
const.SOURCE_TYPE_USB = 3
const.SOURCE_TYPE_GIGE = 4

# Serial port params
const.SERIAL_PORT_STOP = 0
const.SERIAL_PORT_IDLE = 1
const.SERIAL_PORT_CHANGE = 2
const.SERIAL_PORT_RUN = 3

const.DATA_HEADER = "55"

# Communication params
const.SEARCHING_HEADER = 1
const.SEARCHING_LENGTH = 2
const.SEARCHING_COMMAND = 3
const.SEARCHING_DATA = 4

# Camera param
const.TRIGGER_MODE_ON = MV_TRIGGER_MODE_ON
const.TRIGGER_MODE_OFF = MV_TRIGGER_MODE_OFF
const.FRAME_RATE_CONTROL_ON = True
const.FRAME_RATE_CONTROL_OFF = False
const.TRIGGER_SOURCE_LINE0 = MV_TRIGGER_SOURCE_LINE0
const.TRIGGER_POLARITY_RISING_EDGE = 0
const.ANTI_SHAKE_TIME = 1
const.GEV = 100
const.EXPOSURE_TIME = 100.0
const.CACHE_CAPACITY = 400

# setting button mask
const.IMAGE_BUTTON_MASK = 1
const.VIDEO_BUTTON_MASK = 2
const.CAMERA_BUTTON_MASK = 4

# V2S information type
const.CLOSE_SERIAL_PORT = 1
const.OPEN_SERIAL_PORT = 2

# D2V information type
const.STATUS_BAR_SHOW = 1
const.CONTROL_BAR_VISIBLE = 2
const.SLIDER_INIT = 3
const.PLAY_BUTTON_TRIGGERED = 4
const.TOTAL_FRAME = 5
const.CURRENT_FRAME = 6
const.SLIDER_VALUE = 7

# V2D information type
const.PLAY_STATUS = 8
const.FASTER_TRIGGER = 9
const.SLOWER_TRIGGER = 10
const.SLIDER_TRIGGER = 11

# 显示帧率
const.DISPLAY_FRAME_RATE = 30

# control model
const.IDLE_MODE = 0
const.MAKE_TABLE_MODE = 1
const.MAKE_TEMPLATE_MODE = 2
const.PREDICT_MODE = 3
const.IO_WRITE_MODE = 4

# 制作偏移表参数
const.PREDICT_DIRECTION_COUNT = 1
const.RELATIVE_OFFSET = 55
const.NEEDLE_GRID_WIDTH = 32
const.NEEDLE_GRID_HIGH = 93
const.SPIN_ANGELS = [126.707959, None, None]
const.IMAGE_SCALE = [1, None, None]
const.SHIFT_TABLE_FILES = ["./Template/shift_table.csv", None, None]

const.WINDOW_X0 = 235
const.WINDOW_X1 = 275
const.WINDOW_Y0 = 175
const.WINDOW_Y1 = 327

const.TOTAL_NEEDLE_GRID_COUNT = 628
