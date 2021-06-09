# Third-party library
import numpy
from PyQt5.QtCore import *

# User-defined library
from Common.const import const


# 界面处理显示数据端口
class DataHandleThread(QThread):
    serial_port_receive_signal = pyqtSignal(list)
    data_flow_receive_signal = pyqtSignal(numpy.ndarray)

    def __init__(self, UI):
        super(DataHandleThread, self).__init__()
        self.super_UI = UI

    def run(self) -> None:
        while True:
            if not self.super_UI.s2v_data_queue.empty():
                self.serial_port_receive_signal.emit(self.super_UI.s2v_data_queue.get())
            if not self.super_UI.d2v_data_queue.empty():
                self.data_flow_receive_signal.emit(self.super_UI.d2v_data_queue.get())


# 界面处理控制数据端口
class ControlHandleThread(QThread):
    status_bar_info_signal = pyqtSignal(str)
    control_bar_visible_signal = pyqtSignal(bool)
    slider_init_signal = pyqtSignal(int)
    play_button_triggered_signal = pyqtSignal(bool)
    slider_value_signal = pyqtSignal(int)
    current_frame_signal = pyqtSignal(str)
    total_frame_signal = pyqtSignal(str)

    def __init__(self, UI):
        super(ControlHandleThread, self).__init__()
        self.super_UI = UI

    def run(self) -> None:
        while True:
            if self.super_UI.vd_control_pipe.poll():
                direction, info_type, data, detail = self.super_UI.vd_control_pipe.recv()
                #
                if info_type == const.STATUS_BAR_SHOW:
                    self.status_bar_info_signal.emit(detail)
                elif info_type == const.CONTROL_BAR_VISIBLE:
                    self.control_bar_visible_signal.emit(data)
                elif info_type == const.SLIDER_INIT:
                    self.slider_init_signal.emit(data)
                elif info_type == const.PLAY_BUTTON_TRIGGERED:
                    self.play_button_triggered_signal.emit(data)
                elif info_type == const.TOTAL_FRAME:
                    self.total_frame_signal.emit(data)
                elif info_type == const.CURRENT_FRAME:
                    self.current_frame_signal.emit(data)
                elif info_type == const.SLIDER_VALUE:
                    self.slider_value_signal.emit(data)

            elif self.super_UI.vs_control_pipe.poll():
                # TODO:数据接口预留
                direction, info_type, data, detail = self.super_UI.vs_control_pipe.recv()
