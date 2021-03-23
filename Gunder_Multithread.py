# Third-party library
import os
import sys
import re
import datetime
import cv2.cv2 as cv2
from ctypes import c_char_p
from multiprocessing import Process, Value, Queue, Manager, freeze_support
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QAbstractItemView, QMessageBox, QFileDialog, QApplication
from PyQt5.QtCore import *

# User-defined library
import utils
import PredictProcess
import IOProcess
from const import const
from ui_bar_detection import Ui_MainWindow
from SettingDialog import SettingDialog
from AboutDialog import AboutDialog
from DataFlowThread import DataFlowThread
from SerialPortThread import SerialPortThread


class MyWindow(QMainWindow, Ui_MainWindow):
    serial_port_send_signal = pyqtSignal(bytes)

    def __init__(self):
        super().__init__()

        self.save_root = os.path.join(os.getcwd(), "temp")
        if not os.path.exists(self.save_root):
            os.mkdir(self.save_root)

        self.setupUi(self)
        self.input_source = None
        self.serial_port = None

        # 数据流处理线程
        self.data_flow_thread = DataFlowThread(self)
        self.data_flow_thread.start()

        # 串口处理线程
        self.serial_port_thread = SerialPortThread(self)
        self.serial_port_thread.start()

        self.test_image = cv2.imread('./Test/R2878.jpg', cv2.IMREAD_GRAYSCALE)

        # 进程参数
        self.io_queue = Queue()
        self.predict_queue = Queue()
        self.io_process_flag = Value('i', 0)
        self.predict_process_flag = Value('i', 0)
        self.image_save_path = Manager().Value(c_char_p, "./temp/images")
        self.sequence_file_name = Manager().Value(c_char_p, "")

        self.io_process = Process(target=IOProcess.io_process, args=(self.io_queue, self.predict_queue, self.io_process_flag, self.image_save_path, ))
        self.predict_process = Process(target=PredictProcess.predict_process, args=(self.predict_queue, self.predict_process_flag, self.sequence_file_name, ))

        # 窗口相关参数
        self.play_bar_widget.setVisible(False)

        # Action栏信号槽绑定
        self.Setting_action.triggered.connect(self.setting_action_trigger)
        self.Save_action.triggered.connect(self.save_action_trigger)
        self.Close_action.triggered.connect(self.close_action_trigger)
        self.About_action.triggered.connect(self.about_action_trigger)
        self.Login_action.triggered.connect(self.login_action_trigger)

        # Video相关按键信号槽绑定
        self.clicked_flag = True
        self.play_push_button.clicked.connect(self.play_action_trigger)
        self.faster_push_button.clicked.connect(self.video_faster_button_clicked)
        self.slower_push_button.clicked.connect(self.video_slower_button_clicked)
        self.video_progress_slider.sliderMoved.connect(self.video_slider_moved)

        # data flow线程的信号槽
        self.data_flow_thread.status_bar_signal.connect(self.status_show)
        self.data_flow_thread.button_icon_changed_signal.connect(self.button_icon_changed)
        self.data_flow_thread.slider_value_signal.connect(self.slider_value_changed)
        self.data_flow_thread.current_frame_signal.connect(self.current_frame_changed)
        self.data_flow_thread.total_frame_signal.connect(self.total_frame_changed)
        self.data_flow_thread.slider_init_signal.connect(self.slider_init)
        self.data_flow_thread.play_bar_widget_visible_signal.connect(self.play_bar_widget_visible_changed)

        # Training相关按键信号槽绑定
        self.model_flag = True
        self.train_push_Button.clicked.connect(lambda: self.model_running("Template.txt"))
        self.predict_push_button.clicked.connect(lambda: self.predict_push_button_clicked("spin001.txt"))

        # Serial port相关按键信号槽绑定
        self.send_push_button.clicked.connect(self.send_data_to_serial_port)
        self.command_line_edit.textChanged.connect(lambda: utils.lowercase_to_uppercase(self.command_line_edit))
        self.data_line_edit.textChanged.connect(lambda: utils.lowercase_to_uppercase(self.data_line_edit))

        command_hex_reg = QRegExp('^[0-9a-fA-F]{1,2}$')
        command_hex_reg_validator = QtGui.QRegExpValidator(self)
        command_hex_reg_validator.setRegExp(command_hex_reg)
        self.command_line_edit.setValidator(command_hex_reg_validator)
        data_hex_reg = QRegExp('^[0-9a-fA-F]{1,}$')
        data_hex_reg_validator = QtGui.QRegExpValidator(self)
        data_hex_reg_validator.setRegExp(data_hex_reg)
        self.data_line_edit.setValidator(data_hex_reg_validator)

        self.serial_port_thread.serial_port_receive_signal.connect(self.receive_data_from_serial_port)

        # Result table属性设置
        self.tableWidget_total.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget_current.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # TODO：测试代码
        self.io_process_flag.value = const.IO_PROCESS_STARTING
        if not self.io_process.is_alive():
            self.io_process.start()

    '''Action bar signal slot callback function'''
    def save_action_trigger(self):
        root_dir = QFileDialog.getExistingDirectory(self, "Open Directory ", '//')
        if root_dir == "":
            self.save_root = os.getcwd()
        else:
            self.save_root = root_dir

    def setting_action_trigger(self):
        setting_dialog = SettingDialog(self)
        setting_dialog.status_bar_signal.connect(self.status_show)
        result = setting_dialog.exec_()
        source = setting_dialog.get_cur_input_source()
        port = setting_dialog.get_cur_serial_port()
        setting_dialog.destroy()
        # TODO: 串口打开后但不点OK出现串口占用(必现)
        # TODO: 串口切换
        self.serial_port = port
        self.serial_port_thread.serial_port_flag = const.SERIAL_PORT_CHANGE
        if result:
            # TODO: 数据流切换输入源
            self.input_source = self.data_flow_thread.input_source_changed(source)

    def login_action_trigger(self):
        QMessageBox.information(self, 'Login window', "Login function will coming soon!", QMessageBox.Ok, QMessageBox.Ok)

    @staticmethod
    def about_action_trigger():
        about_dialog = AboutDialog()
        _ = about_dialog.exec_()
        about_dialog.destroy()

    # 关闭主程序窗口
    def close_action_trigger(self):
        self.retreat_safely()
        Quit_App = QApplication.instance()
        Quit_App.quit()

    '''Data Flow interface signal slot callback function'''
    def slider_init(self, max_count):
        self.video_progress_slider.setMaximum(max_count)
        self.video_progress_slider.setMinimum(0)
        self.video_progress_slider.setSingleStep(1)

    def play_bar_widget_visible_changed(self, visible):
        self.play_bar_widget.setVisible(visible)

    def current_frame_changed(self, text):
        self.cur_frame_label.setText(text)

    def total_frame_changed(self, text):
        self.total_frame_label.setText(text)

    def slider_value_changed(self, count):
        self.video_progress_slider.setValue(count)

    def button_icon_changed(self, flag):
        play_icon = QtGui.QIcon()
        play_icon.addPixmap(QtGui.QPixmap(":/icon/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.play_push_button.setIcon(play_icon)
        self.clicked_flag = flag

    def status_show(self, show_text):
        self.statusbar.showMessage(show_text, msecs=5000)

    '''Model train/predict/compare signal slot callback function'''
    def model_running(self, sequence_save_file_name):
        print(sequence_save_file_name)
        if self.train_push_Button.isChecked():
            self.sequence_file_name.value = sequence_save_file_name

            # 启动IO线程
            self.io_process_flag.value = const.IO_PROCESS_STARTING
            if not self.io_process.is_alive():
                self.io_process.start()
            # 启动预测线程
            self.predict_process_flag.value = const.PREDICT_PROCESS_STARTING
            if not self.predict_process.is_alive():
                self.predict_process.start()
            self.play_action_trigger()

        else:
            self.play_action_trigger()
            self.io_process_flag.value = const.IO_PROCESS_STOPPING
            self.predict_process_flag.value = const.PREDICT_PROCESS_STOPPING

    def predict_push_button_clicked(self, sequence_save_file_name):
        if self.model_flag:

            self.sequence_file_name.value = sequence_save_file_name
            self.io_process_flag.value = const.IO_PROCESS_STARTING
            self.predict_process_flag.value = const.PREDICT_PROCESS_STARTING
            self.play_action_trigger()
            self.model_flag = False
        else:
            self.play_action_trigger()
            self.io_process_flag.value = const.IO_PROCESS_STOPPING
            self.predict_process_flag.value = const.PREDICT_PROCESS_STOPPING

    '''Video signal slot callback function'''
    # 视频播放信号槽回调函数
    def play_action_trigger(self):
        if self.clicked_flag:
            pause_icon = QtGui.QIcon()
            pause_icon.addPixmap(QtGui.QPixmap(":/icon/pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.play_push_button.setIcon(pause_icon)
            self.clicked_flag = False
            self.data_flow_thread.video_play_flag = True
            self.data_flow_thread.start()
        else:
            play_icon = QtGui.QIcon()
            play_icon.addPixmap(QtGui.QPixmap(":/icon/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.play_push_button.setIcon(play_icon)
            self.clicked_flag = True
            self.data_flow_thread.video_play_flag = False

    # 视频快进信号槽回调函数
    def video_faster_button_clicked(self):
        self.data_flow_thread.video_control_flag = const.VIDEO_FASTER_CLICKED_FLAG

    # 视频快退信号槽回调函数
    def video_slower_button_clicked(self):
        self.data_flow_thread.video_control_flag = const.VIDEO_SLOWER_CLICKED_FLAG

    # 视频滑条信号槽回调函数
    def video_slider_moved(self):
        self.data_flow_thread.video_control_flag = const.VIDEO_SLIDER_MOVED_FLAG

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        print('close')
        self.retreat_safely()

    def retreat_safely(self):
        # Camera安全退出
        # TODO：如果没有选Camera就会奔溃(必现BUG)
        if self.data_flow_thread.source_type in [const.SOURCE_TYPE_VIDEO, const.SOURCE_TYPE_USB, const.SOURCE_TYPE_GIGE]:
            self.data_flow_thread.data_flow_flag = const.DATA_FLOW_RELEASE
        # 线程安全推出
        if self.io_process.is_alive():
            self.io_process.terminate()
        if self.predict_process.is_alive():
            self.predict_process.terminate()

    def send_data_to_serial_port(self):
        try:
            if self.serial_port_thread.serial_port:
                command_length = len(self.command_line_edit.text())
                data_length = len(self.data_line_edit.text())
                if self.command_line_edit.text() and self.data_line_edit.text() and \
                        not command_length % 2 and not data_length % 2:
                    # 所有数据都是str形式
                    head = const.DATA_HEADER
                    length = utils.filling(utils.split_hex(hex((command_length + data_length) // 2 + 1)), 2)
                    data = self.command_line_edit.text() + self.data_line_edit.text()
                    total_data = head + length.upper() + data
                    self.serial_port_send_signal.emit(bytes.fromhex(total_data))
                    pattern = re.compile('.{2}')
                    final_data = ' '.join(pattern.findall(total_data))
                    self.serial_port_data_text_browser.append(utils.colourful_text("[Send]-[{}]-[{}]".format(datetime.datetime.now().strftime('%H:%M:%S.%f'),
                                                                                                             final_data), "red"))
            else:
                QMessageBox.warning(self, 'Serial Port', "Open serial port before sending message", QMessageBox.Ok, QMessageBox.Ok)
        except Exception as error:
            print(error)

    def receive_data_from_serial_port(self, data_list):
        pattern = re.compile('.{2}')
        final_data = ' '.join(pattern.findall(data_list[0]))
        self.serial_port_data_text_browser.append(utils.colourful_text("[Receive]-[{}]-[{}]-[{}]".format(datetime.datetime.now().strftime('%H:%M:%S.%f'),
                                                                                                         final_data.upper(), data_list[1]), "blue"))


if __name__ == '__main__':
    freeze_support()
    app = QApplication(sys.argv)
    ex = MyWindow()
    ex.show()
    sys.exit(app.exec_())
