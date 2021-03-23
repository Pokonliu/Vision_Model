from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from const import const
from ui_setting_dialog import Ui_dialog
from SearchDeviceThread import SearchDeviceThread
from SerialPortThread import Communication


class SettingDialog(QDialog, Ui_dialog):
    status_bar_signal = pyqtSignal(str)

    def __init__(self, UI):
        super().__init__()
        self.setupUi(self)

        '''Params definition'''
        self.super_UI = UI
        self.serial_port = self.super_UI.serial_port
        self.search_device_thread = SearchDeviceThread(self)

        '''Input Source Setting'''
        self.image_radio_button.toggled.connect(self.image_radio_button_toggled)
        self.image_tool_button.clicked.connect(self.image_tool_button_clicked)

        self.video_radio_button.toggled.connect(self.video_radio_button_toggled)
        self.video_tool_button.clicked.connect(self.video_tool_button_clicked)

        self.camera_radio_button.toggled.connect(self.camera_radio_button_toggled)
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

        self.source_output_label.setText(self.super_UI.input_source)
        if self.serial_port:
            self.serial_output_label.setText(self.serial_port.name())
            self.baud_rate_combo_box.setCurrentText(str(self.serial_port.com.baudrate))
            self.byte_size_combo_box.setCurrentText(str(self.serial_port.com.bytesize))
            self.stop_bits_combo_box.setCurrentText(str(self.serial_port.com.stopbits))
        self.check_serial_port()

    def finish_show(self):
        self.status_bar_signal.emit("New devices has been discovered")

    def search_device_push_button_clicked(self):
        self.status_bar_signal.emit("Searching for new devices...")
        self.search_device_thread.start()

    def image_radio_button_toggled(self):
        self.image_tool_button.setEnabled(True)
        self.video_tool_button.setEnabled(False)
        self.camera_combo_box.setEnabled(False)

    def video_radio_button_toggled(self):
        self.image_tool_button.setEnabled(False)
        self.video_tool_button.setEnabled(True)
        self.camera_combo_box.setEnabled(False)

    def camera_radio_button_toggled(self):
        self.image_tool_button.setEnabled(False)
        self.video_tool_button.setEnabled(False)
        self.camera_combo_box.setEnabled(True)

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
        serial_ports = Communication.com_check()
        for port_info in serial_ports:
            self.serial_port_combo_box.addItem(port_info)

    def open_serial_push_button_clicked(self):
        cur_com = self.serial_port_combo_box.currentText()
        baud_rate = self.baud_rate_combo_box.currentText()
        byte_size = int(self.byte_size_combo_box.currentText())
        stop_bits = int(self.stop_bits_combo_box.currentText())
        try:
            self.serial_port = Communication(cur_com, baud_rate, byte_size, stop_bits)
            if self.serial_port.com.isOpen():
                self.serial_output_label.setText(cur_com)
                self.control_set_enable(False)
            else:
                raise Exception("open failed")
        except Exception as error:
            QMessageBox.critical(self, 'Serial Port Error', "Serial port is occupied!", QMessageBox.Ok, QMessageBox.Ok)
            print("Serial port init occurred {}".format(error))

    def close_serial_push_button_clicked(self):
        try:
            # 先保证线程处于idle的阶段再停止
            self.super_UI.serial_port_thread.serial_port_flag = const.SERIAL_PORT_IDLE
            self.serial_port.close()
            if not self.serial_port.com.isOpen():
                self.serial_output_label.setText("")
                self.control_set_enable(True)
            else:
                raise Exception("close failed")
        except Exception as error:
            print("Serial port release occurred {}".format(error))
        finally:
            self.serial_port = None

    def led_brightness_slider_value_changed(self):
        # TODO 串口下发LED亮度参数
        pass

    def get_cur_serial_port(self):
        return self.serial_port

    def check_serial_port(self):
        if self.serial_port:
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
