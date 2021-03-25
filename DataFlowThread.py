import os
import time
import cv2.cv2 as cv2
import numpy as np
import utils
from const import const
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from MvImport.MvCameraControl_class import *


class DataFlowThread(QThread):
    status_bar_signal = pyqtSignal(str)
    slider_value_signal = pyqtSignal(int)
    slider_init_signal = pyqtSignal(int)
    current_frame_signal = pyqtSignal(str)
    total_frame_signal = pyqtSignal(str)
    button_icon_changed_signal = pyqtSignal(bool)
    play_bar_widget_visible_signal = pyqtSignal(bool)

    def __init__(self, UI):
        super().__init__()
        self.super_UI = UI
        self.input_source = ""
        self.source_changed_flag = False
        self.source_type = const.SOURCE_TYPE_NONE
        self.data_flow_flag = const.DATA_FLOW_IDLE

        '''Camera params'''
        # USB params
        self.USB_camera = None
        # GigE params:
        self.GigE_camera = None
        self.nPayloadSize = None
        self.nPixelFormat = None
        self.data_buf = None

        self.frame_count = 0

        '''Video params'''
        self.video_src = None
        self.frame_single_step = const.VIDEO_FRAME_SINGLE_STEP
        self.video_control_flag = const.VIDEO_CONTROL_NONE
        self.video_play_flag = const.VIDEO_STOP
        self.total_frame_num = 0
        self.cur_frame_num = 0
        self.video_high = 0
        self.video_width = 0

        '''Test params'''
        # TODO: 测试完成后期删除
        self.needle_direction = 0
        self.needle_row = 0
        self.needle_col = 0

    def run(self):
        data_flow_function_point = {const.DATA_FLOW_IDLE: self.data_flow_idle,
                                    const.DATA_FLOW_CHANGE: self.data_flow_change,
                                    const.DATA_FLOW_INIT: self.data_flow_init,
                                    const.DATA_FLOW_RUN: self.data_flow_run,
                                    const.DATA_FLOW_RELEASE: self.data_flow_release}
        while self.data_flow_flag:
            data_flow_function_point.get(self.data_flow_flag)()

    def input_source_changed(self, source):
        # 首次更新直接进入Change
        if not self.input_source:
            self.data_flow_flag = const.DATA_FLOW_CHANGE
        # source发生了切换，且之前的source为Video or Camera(USB\GigE) → 先进入Release再进入Change
        elif self.input_source != source:
            if self.source_type in [const.SOURCE_TYPE_VIDEO, const.SOURCE_TYPE_USB, const.SOURCE_TYPE_GIGE]:
                self.source_changed_flag = True
                self.data_flow_flag = const.DATA_FLOW_RELEASE
            else:
                self.data_flow_flag = const.DATA_FLOW_CHANGE
        self.input_source = source
        return self.input_source

    @staticmethod
    def data_flow_idle():
        time.sleep(0.01)

    def data_flow_change(self):
        if self.input_source == "":
            self.status_bar_signal.emit("You didn't choose anything")
            self.source_type = const.SOURCE_TYPE_NONE
            # 直接进入空闲状态
            self.data_flow_flag = const.DATA_FLOW_IDLE
        elif os.path.exists(self.input_source) and self.input_source.endswith((".jpg", ".gif", ".png", ".bmp")):
            self.status_bar_signal.emit("Input source changed to Image")
            self.source_type = const.SOURCE_TYPE_IMAGE
            self.data_flow_flag = const.DATA_FLOW_INIT
        elif os.path.exists(self.input_source) and self.input_source.endswith((".avi", ".mp4", ".wmv", ".rmvb")):
            self.status_bar_signal.emit("Input source changed to Video")
            self.source_type = const.SOURCE_TYPE_VIDEO
            self.data_flow_flag = const.DATA_FLOW_INIT
        elif self.input_source.split()[0] == "USB":
            self.status_bar_signal.emit("Input source changed to USB {}".format(self.input_source.split()[1]))
            self.source_type = const.SOURCE_TYPE_USBw
            self.data_flow_flag = const.DATA_FLOW_INIT
        elif self.input_source.split()[0] == "GigE":
            self.status_bar_signal.emit("Input source changed to GigE {}".format(self.input_source.split()[1]))
            self.source_type = const.SOURCE_TYPE_GIGE
            self.data_flow_flag = const.DATA_FLOW_INIT

    def data_flow_init(self):
        init_function_point = {const.SOURCE_TYPE_IMAGE: self.image_source_init,
                               const.SOURCE_TYPE_VIDEO: self.video_source_init,
                               const.SOURCE_TYPE_USB: self.USB_source_init,
                               const.SOURCE_TYPE_GIGE: self.GigE_source_init}
        init_function_point.get(self.source_type)()

    def data_flow_run(self):
        run_function_point = {const.SOURCE_TYPE_VIDEO: self.video_source_run,
                              const.SOURCE_TYPE_USB: self.USB_source_run,
                              const.SOURCE_TYPE_GIGE: self.GigE_source_run}
        run_function_point.get(self.source_type)()

    def data_flow_release(self):
        release_function_point = {const.SOURCE_TYPE_VIDEO: self.video_source_release,
                                  const.SOURCE_TYPE_USB: self.USB_source_release,
                                  const.SOURCE_TYPE_GIGE: self.GigE_source_release}
        release_function_point.get(self.source_type)()

    '''source init function'''
    def image_source_init(self):
        self.play_bar_widget_visible_signal.emit(False)
        src = cv2.imread(self.input_source, cv2.IMREAD_COLOR)
        self.show_image_to_label(src, self.super_UI.input_label, const.BGR2RGB)
        # 初始化完成进入空闲状态
        self.data_flow_flag = const.DATA_FLOW_IDLE

    def video_source_init(self):
        # 当前source是视频流路径，初始化视频流对象
        self.video_src = cv2.VideoCapture(self.input_source)

        # 视频参数初始化
        self.total_frame_num = int(self.video_src.get(7))
        self.video_width = int(self.video_src.get(3))
        self.video_high = int(self.video_src.get(4))

        # 主窗口的界面显示
        self.play_bar_widget_visible_signal.emit(True)

        # 进度条初始化设置
        self.slider_init_signal.emit(self.total_frame_num)

        # 播放按键初始化
        self.button_icon_changed_signal.emit(True)

        # input label显示第一帧图像
        try:
            ret, frame = self.video_src.read()
            if not ret:
                raise Exception("读取第一帧失败")
            else:
                self.show_image_to_label(frame, self.super_UI.input_label, const.BGR2RGB)

        except Exception as error:
            print("读取视频流出现错误：{}".format(error))

        # 进度条初始化设置
        self.slider_value_signal.emit(0)

        # 帧数label初始化设置
        self.cur_frame_num = 1
        self.total_frame_signal.emit(str(self.total_frame_num))
        self.current_frame_signal.emit(str(self.cur_frame_num))

        # 将执行状态转移到run
        self.data_flow_flag = const.DATA_FLOW_RUN

    def USB_source_init(self):
        self.play_bar_widget_visible_signal.emit(False)
        self.USB_camera = cv2.VideoCapture(int(self.input_source.split()[1]), cv2.CAP_DSHOW)
        # 将执行状态转移到run
        self.data_flow_flag = const.DATA_FLOW_RUN

    def GigE_source_init(self):
        self.play_bar_widget_visible_signal.emit(False)
        device_list = MV_CC_DEVICE_INFO_LIST()
        device_type = MV_GIGE_DEVICE | MV_USB_DEVICE

        # 枚举设备
        ret = MvCamera.MV_CC_EnumDevices(device_type, device_list)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)

        print("Find %d devices!" % device_list.nDeviceNum)

        # 创建相机实例
        self.GigE_camera = MvCamera()

        # 选择设备并创建句柄
        stDeviceList = cast(device_list.pDeviceInfo[int(self.input_source.split()[1])], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.GigE_camera.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)

        # 打开设备
        ret = self.GigE_camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)

        # 探测网络最佳包大小(只对GigE相机有效)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.GigE_camera.MV_CC_GetOptimalPacketSize()
            print('PacketSize:', nPacketSize)
            if int(nPacketSize) > 0:
                ret = self.GigE_camera.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # 设置触发模式为ON
        ret = self.GigE_camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)

        # # 设置触发源为Line0
        # ret = self.GigE_camera.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0)
        # if ret != 0:
        #     print("set trigger source fail! ret[0x%x]" % ret)
        #
        # # 设置触发源为上升沿
        # ret = self.GigE_camera.MV_CC_SetEnumValue("TriggerActivation", 0)
        # if ret != 0:
        #     print("set trigger activation fail! ret[0x%x]" % ret)
        #
        # # 设置线路防抖时间为1us
        # ret = self.GigE_camera.MV_CC_SetIntValue("LineDebouncerTime", 1)
        # if ret != 0:
        #     print("set line debouncer time fail! ret[0x%x]" % ret)

        # 设置 GEV SCPD值来满足最大最大帧率运行
        ret = self.GigE_camera.MV_CC_SetIntValue("GevSCPD", 400)
        if ret != 0:
            print("set line Gev SCPD fail! ret[0x%x]" % ret)

        # 设置曝光时间
        ret = self.GigE_camera.MV_CC_SetFloatValue("ExposureTime", 50.0)
        if ret != 0:
            print("set line Exposure Time fail! ret[0x%x]" % ret)

        # 获取数据包大小
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.GigE_camera.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)

        self.nPayloadSize = stParam.nCurValue

        # 获取像素格式
        stEnumParam = MVCC_ENUMVALUE()
        ret = self.GigE_camera.MV_CC_GetEnumValue("PixelFormat", stEnumParam)
        if ret != 0:
            print("get pixel format fail! ret[0x%x]" % ret)
        self.nPixelFormat = stEnumParam.nCurValue

        # 开始取流
        ret = self.GigE_camera.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)
        # 将执行状态转移到run
        self.data_flow_flag = const.DATA_FLOW_RUN

    '''source run function'''
    def video_source_run(self):
        try:
            if self.video_play_flag or self.video_control_management():
                if self.video_play_flag:
                    self.video_control_management()
                ret, frame = self.video_src.read()
                if ret:
                    self.show_image_to_label(frame, self.super_UI.input_label, const.BGR2RGB)

                    # 后期落地可以删除
                    # transform_element = cv2.getRotationMatrix2D(center=(self.video_width / 2, self.video_high / 2), angle=const.TOP_SIDE_ANGLE, scale=const.SCALE_RATIO)
                    # spin_image = cv2.warpAffine(frame, transform_element, (self.video_width, self.video_high), borderValue=const.FILLING_COLOR)
                    # if self.super_UI.predict_process_flag.value == const.PREDICT_PROCESS_STARTING:
                    #     self.super_UI.predict_queue.put(spin_image)

                    # for i in range(const.CROP_COUNT):
                    #     cv2.rectangle(img=spin_image, pt1=(const.STARTING_COORDINATES[0], const.STARTING_COORDINATES[1]),
                    #                   pt2=(const.STARTING_COORDINATES[0] + const.NEEDLE_GRID_WIDTH * (i + 1), const.STARTING_COORDINATES[1] + 100), color=(255, 100, 100), thickness=1)
                    # self.show_image_to_label(spin_image, self.super_UI.output_label, const.BGR2RGB)
                else:
                    self.data_flow_flag = const.DATA_FLOW_RELEASE
                self.cur_frame_num += 1
                self.current_frame_signal.emit(str(self.cur_frame_num))
                self.slider_value_signal.emit(self.cur_frame_num)
            time.sleep(0.01)
        # 如果程序异常则强制进行Video相关资源的释放
        except Exception as er:
            print("Error appear as {}".format(er))
            self.data_flow_flag = const.DATA_FLOW_RELEASE

    def USB_source_run(self):
        ret, frame = self.USB_camera.read()
        if ret:
            self.show_image_to_label(frame, self.super_UI.input_label, const.BGR2RGB)

    def GigE_source_run(self):
        # 获取当前采集帧率来处理显示帧率
        cur_frame = MVCC_FLOATVALUE()
        ret = self.GigE_camera.MV_CC_GetFloatValue("ResultingFrameRate", cur_frame)
        if ret != 0:
            print("Get result frame rate fail! ret[0x%x]" % ret)
        # 读取数据流
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        self.data_buf = (c_ubyte * self.nPayloadSize)()
        ret = self.GigE_camera.MV_CC_GetOneFrameTimeout(self.data_buf, self.nPayloadSize, stFrameInfo, 1000)
        if ret == 0:
            if 0x01080001 == self.nPixelFormat:
                data_mono_arr = np.array(self.data_buf).reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
                # 测试图片程序入口
                # self.super_UI.io_queue.put([data_mono_arr, self.needle_direction, self.needle_row, self.needle_col])
                # 预测程序入口
                if self.super_UI.predict_process_flag.value:
                    # TODO: 网络为3通道input，将单通道图像堆叠为三通道输入，后期移除
                    image_data = np.stack((data_mono_arr, data_mono_arr, data_mono_arr), axis=-1)
                    self.super_UI.predict_queue.put([image_data, self.needle_direction, self.needle_row, self.needle_col])

                # TODO: 显示帧率设置为30帧防止界面卡死
                if self.frame_count > cur_frame.fCurValue // const.RESULTING_FRAME_RATE:
                    self.show_image_to_label(data_mono_arr, self.super_UI.input_label, const.GRAY2RGB)
                    self.frame_count = 0
                else:
                    self.frame_count += 1

            # TODO:目前只处理灰度图像，RGB图像后期加入
            if 0x02180014 == self.nPixelFormat:
                data_mono_arr = np.array(self.data_buf)

                data_r = data_mono_arr[0:stFrameInfo.nFrameLen:3]
                data_g = data_mono_arr[1:stFrameInfo.nFrameLen:3]
                data_b = data_mono_arr[2:stFrameInfo.nFrameLen:3]

                data_r_arr = data_r.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
                data_g_arr = data_g.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
                data_b_arr = data_b.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)

                RGB_data = np.stack([data_r_arr, data_g_arr, data_b_arr], axis=-1)
                self.show_image_to_label(RGB_data, self.super_UI.input_label)

    '''source release function'''
    def video_source_release(self):
        self.video_src.release()
        self.video_src = None
        # 中途切换视频退出
        if self.source_changed_flag:
            self.video_play_flag = const.VIDEO_STOP
            self.data_flow_flag = const.DATA_FLOW_CHANGE
        # 自动播放到结束退出 或 中途出现异常
        else:
            # 自动播放到结束退出
            if self.cur_frame_num >= self.total_frame_num:
                self.video_play_flag = const.VIDEO_STOP
                self.data_flow_flag = const.DATA_FLOW_INIT
            # TODO 视情况应该由上层来处理
            # 视频流出现错误退出
            else:
                # TODO 具体需要复现场景Debug
                self.video_play_flag = const.VIDEO_STOP
                self.data_flow_flag = const.DATA_FLOW_INIT

    def USB_source_release(self):
        self.USB_camera.release()
        self.USB_camera = None
        self.data_flow_flag = const.DATA_FLOW_CHANGE

    def GigE_source_release(self):
        # 停止取流
        ret = self.GigE_camera.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)
            self.data_buf = None

        # 关闭设备
        ret = self.GigE_camera.MV_CC_CloseDevice()
        if ret != 0:
            print("close device fail! ret[0x%x]" % ret)
            self.data_buf = None

        # 销毁句柄
        ret = self.GigE_camera.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)
            self.data_buf = None

        self.GigE_camera = None
        self.data_buf = None
        self.data_flow_flag = const.DATA_FLOW_CHANGE

    def video_control_management(self):
        # 快进、快退、滑动控制，注意边缘处理
        if self.video_control_flag:
            if self.video_control_flag in [const.VIDEO_SLOWER_CLICKED_FLAG, const.VIDEO_FASTER_CLICKED_FLAG]:
                self.cur_frame_num += (self.frame_single_step * self.video_control_flag)
                self.cur_frame_num = max(self.cur_frame_num, 0)
                self.cur_frame_num = min(self.cur_frame_num, self.total_frame_num)
            elif self.video_control_flag == const.VIDEO_SLIDER_MOVED_FLAG:
                self.cur_frame_num = self.super_UI.video_progress_slider.value()
            self.video_src.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_num)
            self.video_control_flag = const.VIDEO_CONTROL_NONE
            self.cur_frame_num -= 1
            return True
        else:
            return False

    @staticmethod
    def show_image_to_label(src, label, flag=None):
        frame_QPixmap = QPixmap.fromImage(QImage(cv2.cvtColor(src, flag) if flag else src, src.shape[1], src.shape[0], QImage.Format_RGB888))
        label.setPixmap(frame_QPixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))
