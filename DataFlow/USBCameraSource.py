# Third-party library
from cv2 import cv2


class USBCamera:
    def __init__(self, device_name):
        self.camera_instance = cv2.VideoCapture(device_name, cv2.CAP_DSHOW)

    def run(self):
        """
        :return:[res, img_data]
        """
        return self.camera_instance.read()

    def release(self):
        self.camera_instance.release()
        self.camera_instance = None
