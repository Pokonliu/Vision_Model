# Third-party library
import cv2.cv2 as cv2

# User-defined library
from PyQt5.QtCore import QThread
from DataFlow.MvImport.MvCameraControl_class import *


class SearchDeviceThread(QThread):
    def __init__(self, UI):
        super().__init__()
        self.super_UI = UI
        self.camera = None

    def run(self):
        self.super_UI.camera_combo_box.clear()
        self.adding_USB_camera_source()
        self.adding_GigE_camera_source()

    def adding_USB_camera_source(self):
        self.super_UI.camera_combo_box.addItem("")
        for i in range(5):
            self.camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if self.camera.isOpened():
                self.super_UI.camera_combo_box.addItem("USB {}".format(i))
            self.camera.release()

    def adding_GigE_camera_source(self):
        device_list = MV_CC_DEVICE_INFO_LIST()
        device_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(device_type, device_list)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)

        print("Find %d devices!" % device_list.nDeviceNum)
        for i in range(0, device_list.nDeviceNum):
            self.super_UI.camera_combo_box.addItem("GigE {}".format(i))
