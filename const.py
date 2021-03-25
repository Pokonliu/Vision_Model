import cv2.cv2 as cv2


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

const.VIDEO_PLAY = 1
const.VIDEO_STOP = 0

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
const.NEEDLE_GRID_WIDTH = 39
const.NEEDLE_GRID_HIGH = 89
const.START_IMAGE_NUMBER_R = -2014
const.START_IMAGE_NUMBER_L = 2128
const.EACH_LINE_COUNTS = 529

# Train params
const.MODEL_PATH = "./stf_new_model/threeLayers.pb"
const.TF_MODEL_PATH = "./stf_new_model"
const.PREDICT_DIRECTION_COUNT = 1
const.PREDICT_ANGELS = [55.21826658383292, None, None]
const.PREDICT_SCALE = [1.21741324376324, None, None]
const.COORDINATE_FILES = ["final.csv", None, None]

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
const.RESULTING_FRAME_RATE = 30
