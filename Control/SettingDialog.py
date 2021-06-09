# Third-party library
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# User-defined library
from Common.const import const
from Control.SearchDeviceThread import SearchDeviceThread
from View.ui_setting_dialog import Ui_dialog
from DataFlow.SerialPort import Communication


class SettingDialog(QDialog, Ui_dialog):
    status_bar_signal = pyqtSignal(str)

    def __init__(self, UI):
        super().__init__()
        self.setupUi(self)

        try:
            '''Params definition'''
            self.super_UI = UI
            self.search_device_thread = SearchDeviceThread(self)

            '''Input Source Setting'''
            self.image_radio_button.toggled.connect(lambda: self.radio_button_toggled(const.IMAGE_BUTTON_MASK))
            self.image_tool_button.clicked.connect(self.image_tool_button_clicked)

            self.video_radio_button.toggled.connect(lambda: self.radio_button_toggled(const.VIDEO_BUTTON_MASK))
            self.video_tool_button.clicked.connect(self.video_tool_button_clicked)

            self.camera_radio_button.toggled.connect(lambda: self.radio_button_toggled(const.CAMERA_BUTTON_MASK))
            self.camera_combo_box.currentIndexChanged.connect(self.camera_combo_box_current_index_changed)

            self.search_device_push_button.clicked.connect(self.search_device_push_button_clicked)
            self.search_device_thread.finished.connect(self.finish_show)

            '''Serial Port Setting'''
            self.search_serial_push_button.clicked.connect(self.search_serial_push_button_clicked)
            self.open_serial_push_button.clicked.connect(self.open_serial_push_button_clicked)
            self.close_serial_push_button.clicked.connect(self.close_serial_push_button_clicked)
            self.led_brightness_slider.valueChanged.connect(self.led_brightness_slider_value_changed)

            self.buttonBox.accepted.connect(self.accept)
            self.buttonBox.rejected.connect(self.reject)

            self.source_output_label.setText(self.super_UI.v2d_params_str.value)
            if self.super_UI.v2s_control_flag.value == const.SERIAL_PORT_RUN:
                self.serial_output_label.setText(self.super_UI.v2s_params_dict["com"])
                self.baud_rate_combo_box.setCurrentText(str(self.super_UI.v2s_params_dict["baud_rate"]))
                self.byte_size_combo_box.setCurrentText(str(self.super_UI.v2s_params_dict["byte_size"]))
                self.stop_bits_combo_box.setCurrentText(str(self.super_UI.v2s_params_dict["stop_bits"]))
            self.check_serial_port()
        except Exception as error:
            print(error)

    def finish_show(self):
        self.status_bar_signal.emit("New devices has been discovered")

    def search_device_push_button_clicked(self):
        self.status_bar_signal.emit("Searching for new devices...")
        self.search_device_thread.start()

    def radio_button_toggled(self, switch):
        self.image_tool_button.setEnabled(switch & 1 == 1)
        self.video_tool_button.setEnabled(switch & 2 == 2)
        self.camera_combo_box.setEnabled(switch & 4 == 4)

    def image_tool_button_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', '//', "Image files (*.jpg *.gif *.png *.bmp)",)
        self.source_output_label.setText(file_name)

    def video_tool_button_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', '//', "Image files (*.avi *.mp4 *.wmv *.rmvb)",)
        self.source_output_label.setText(file_name)

    def camera_combo_box_current_index_changed(self):
        self.source_output_label.setText(self.camera_combo_box.currentText())

    def get_cur_input_source(self):
        return self.source_output_label.text()

    def search_serial_push_button_clicked(self):
        self.serial_port_combo_box.clear()
        serial_ports = Communication.com_check()
        for port_info in serial_ports:
            self.serial_port_combo_box.addItem(port_info)

    def open_serial_push_button_clicked(self):
        try:
            self.super_UI.v2s_params_dict["com"] = self.serial_port_combo_box.currentText()
            self.super_UI.v2s_params_dict["baud_rate"] = self.baud_rate_combo_box.currentText()
            self.super_UI.v2s_params_dict["byte_size"] = int(self.byte_size_combo_box.currentText())
            self.super_UI.v2s_params_dict["stop_bits"] = int(self.stop_bits_combo_box.currentText())

            self.super_UI.v2s_control_flag.value = const.SERIAL_PORT_CHANGE
            # TODO:此处需要等待串口线程返回相应(实际可能会卡死，待测试), 数据通道需要继续优化
            direction, info_type, res, command = self.super_UI.vs_control_pipe.recv()
            if res:
                self.serial_output_label.setText(self.serial_port_combo_box.currentText())
                self.control_set_enable(False)
            else:
                raise Exception("open failed")
        except Exception as error:
            QMessageBox.critical(self, 'Serial Port Error', "Serial port is occupied!", QMessageBox.Ok, QMessageBox.Ok)
            print("Serial port init occurred {}".format(error))

    def close_serial_push_button_clicked(self):
        try:
            # 只用于通知
            self.super_UI.vs_control_pipe.send(["V2S", const.CLOSE_SERIAL_PORT, None, None])
            # 等待接收
            direction, info_type, res, command = self.super_UI.vs_control_pipe.recv()
            if res:
                self.serial_output_label.setText("")
                self.control_set_enable(True)
            else:
                raise Exception("close failed")
        except Exception as error:
            print("Serial port release occurred {}".format(error))

    # TODO 串口下发LED亮度参数
    def led_brightness_slider_value_changed(self):
        pass

    def check_serial_port(self):
        if self.super_UI.v2s_control_flag.value == const.SERIAL_PORT_RUN:
            self.control_set_enable(False)
        else:
            self.control_set_enable(True)

    def control_set_enable(self, condition):
        self.serial_port_combo_box.setEnabled(condition)
        self.baud_rate_combo_box.setEnabled(condition)
        self.byte_size_combo_box.setEnabled(condition)
        self.stop_bits_combo_box.setEnabled(condition)
        self.open_serial_push_button.setEnabled(condition)
        self.close_serial_push_button.setEnabled(not condition)
        self.led_brightness_slider.setEnabled(not condition)
