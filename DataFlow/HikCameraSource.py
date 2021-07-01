# Third-party library
import numpy as np

# User-defined library
from DataFlow.MvImport.MvCameraControl_class import *


# 海康相机基类
class HikCamera:
    # 初始化只传入设备串信息
    def __init__(self, device_name, trigger_mode, frame_rate_control, trigger_source, trigger_polarity, anti_shake_time, GEV, exposure_time, cache_capacity):
        self.camera_instance = None
        self.data_buf = None
        self.frame_info = None
        self.payload_size = None
        self.pixel_format = None
        self.img_data = None
        self.GigE_source_init(device_name, trigger_mode, frame_rate_control, trigger_source, trigger_polarity, anti_shake_time, GEV, exposure_time, cache_capacity)

    def GigE_source_init(self, device_name, trigger_mode, frame_rate_control, trigger_source, trigger_polarity, anti_shake_time, GEV, exposure_time, cache_capacity):
        deviceList = MV_CC_DEVICE_INFO_LIST()
        deviceType = MV_GIGE_DEVICE | MV_USB_DEVICE

        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(deviceType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            print("find no device!")
            sys.exit()

        print("Find %d devices!" % deviceList.nDeviceNum)

        # 创建相机实例
        self.camera_instance = MvCamera()

        # 选择设备并创建句柄
        stDeviceList = cast(deviceList.pDeviceInfo[device_name], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.camera_instance.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            self.camera_instance.MV_CC_DestroyHandle()
            print("create handle fail! ret[0x%x]" % ret)

        # 打开设备
        ret = self.camera_instance.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)

        print("open device successfully!")

        # 探测网络最佳包大小(只对GigE相机有效)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.camera_instance.MV_CC_GetOptimalPacketSize()
            print('PacketSize:', nPacketSize)
            if int(nPacketSize) > 0:
                ret = self.camera_instance.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # 设置触发模式为ON
        ret = self.camera_instance.MV_CC_SetEnumValue("TriggerMode", trigger_mode)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)

        # 关闭帧率控制
        ret = self.camera_instance.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", c_bool(frame_rate_control))
        if ret != 0:
            print("Set acquisition frame rate fail! ret[0x%x]" % ret)

        # 设置触发源为Line0
        ret = self.camera_instance.MV_CC_SetEnumValue("TriggerSource", trigger_source)
        if ret != 0:
            print("set trigger source fail! ret[0x%x]" % ret)

        # 设置触发源为上升沿
        ret = self.camera_instance.MV_CC_SetEnumValue("TriggerActivation", trigger_polarity)
        if ret != 0:
            print("set trigger activation fail! ret[0x%x]" % ret)

        # 设置线路防抖时间为1us
        ret = self.camera_instance.MV_CC_SetIntValue("LineDebouncerTime", anti_shake_time)
        if ret != 0:
            print("set line debouncer time fail! ret[0x%x]" % ret)

        # 设置 GEV SCPD值来满足最大最大帧率运行
        ret = self.camera_instance.MV_CC_SetIntValue("GevSCPD", GEV)
        if ret != 0:
            print("set line Gev SCPD fail! ret[0x%x]" % ret)

        # 设置曝光时间
        ret = self.camera_instance.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        if ret != 0:
            print("set line Exposure Time fail! ret[0x%x]" % ret)

        # 设置图像缓存节点数量为400
        ret = self.camera_instance.MV_CC_SetImageNodeNum(cache_capacity)
        if ret != 0:
            print("set image node num fail! ret[0x%x]" % ret)

        # 获取数据包大小
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.camera_instance.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)

        self.payload_size = stParam.nCurValue

        # 获取像素格式
        stEnumParam = MVCC_ENUMVALUE()
        ret = self.camera_instance.MV_CC_GetEnumValue("PixelFormat", stEnumParam)
        if ret != 0:
            print("get pixel format fail! ret[0x%x]" % ret)
        self.pixel_format = stEnumParam.nCurValue

        # 开始取流
        ret = self.camera_instance.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)

        # 数据流预处理
        self.frame_info = MV_FRAME_OUT_INFO_EX()
        memset(byref(self.frame_info), 0, sizeof(self.frame_info))
        self.data_buf = (c_ubyte * self.payload_size)()

    def run(self):
        # 读取数据流
        ret = self.camera_instance.MV_CC_GetOneFrameTimeout(self.data_buf, self.payload_size, self.frame_info, 1000)
        if ret == 0:
            # 处理灰度图像
            if 0x01080001 == self.pixel_format:
                self.img_data = np.array(self.data_buf).reshape(self.frame_info.nHeight, self.frame_info.nWidth)
            # if self.Is_mono_data(self.pixel_format):
            #     self.img_data = self.Mono_numpy(self.data_buf, self.frame_info.nWidth, self.frame_info.nHeight)
            # 处理彩色图像
            elif self.Is_color_data(self.pixel_format):
                self.img_data = self.Color_numpy(self.data_buf, self.frame_info.nWidth, self.frame_info.nHeight)
        return not ret, self.img_data

    def release(self):
        # 停止取流
        ret = self.camera_instance.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)

        # 关闭设备
        ret = self.camera_instance.MV_CC_CloseDevice()
        if ret != 0:
            print("close device fail! ret[0x%x]" % ret)

        # 销毁句柄
        ret = self.camera_instance.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)

        print("close device successfully!")
        self.camera_instance = None
        self.data_buf = None
        self.frame_info = None
        self.payload_size = None
        self.pixel_format = None
        self.img_data = None

    @staticmethod
    def Is_mono_data(PixelType):
        if PixelType_Gvsp_Mono8 == PixelType or PixelType_Gvsp_Mono10 == PixelType \
                or PixelType_Gvsp_Mono10_Packed == PixelType or PixelType_Gvsp_Mono12 == PixelType \
                or PixelType_Gvsp_Mono12_Packed == PixelType:
            return True
        else:
            return False

    @staticmethod
    def Is_color_data(PixelType):
        if PixelType_Gvsp_BayerGR8 == PixelType or PixelType_Gvsp_BayerRG8 == PixelType \
                or PixelType_Gvsp_BayerGB8 == PixelType or PixelType_Gvsp_BayerBG8 == PixelType \
                or PixelType_Gvsp_BayerGR10 == PixelType or PixelType_Gvsp_BayerRG10 == PixelType \
                or PixelType_Gvsp_BayerGB10 == PixelType or PixelType_Gvsp_BayerBG10 == PixelType \
                or PixelType_Gvsp_BayerGR12 == PixelType or PixelType_Gvsp_BayerRG12 == PixelType \
                or PixelType_Gvsp_BayerGB12 == PixelType or PixelType_Gvsp_BayerBG12 == PixelType \
                or PixelType_Gvsp_BayerGR10_Packed == PixelType or PixelType_Gvsp_BayerRG10_Packed == PixelType \
                or PixelType_Gvsp_BayerGB10_Packed == PixelType or PixelType_Gvsp_BayerBG10_Packed == PixelType \
                or PixelType_Gvsp_BayerGR12_Packed == PixelType or PixelType_Gvsp_BayerRG12_Packed == PixelType \
                or PixelType_Gvsp_BayerGB12_Packed == PixelType or PixelType_Gvsp_BayerBG12_Packed == PixelType \
                or PixelType_Gvsp_YUV422_Packed == PixelType or PixelType_Gvsp_YUV422_YUYV_Packed == PixelType:
            return True
        else:
            return False

    @staticmethod
    def Mono_numpy(data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
        data_arr = data_.reshape(nHeight, nWidth)
        return data_arr

    @staticmethod
    def Color_numpy(data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight * 3), dtype=np.uint8, offset=0)

        data_r = data_[0: nWidth * nHeight: 3]
        data_g = data_[1: nWidth * nHeight: 3]
        data_b = data_[2: nWidth * nHeight: 3]

        data_r_arr = data_r.reshape(nHeight, nWidth)
        data_g_arr = data_g.reshape(nHeight, nWidth)
        data_b_arr = data_b.reshape(nHeight, nWidth)

        data_arr = np.stack([data_r_arr, data_g_arr, data_b_arr], axis=-1)
        return data_arr
