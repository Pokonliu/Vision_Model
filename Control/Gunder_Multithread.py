# Third-party library
import os
import re
import time
import datetime
from ctypes import c_char_p
from multiprocessing import Value, Queue, Manager, Pipe
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# User-defined library
from Common import utils
from Common.const import const
from View.ui_main_window import Ui_MainWindow
from Control.SettingDialog import SettingDialog
from Control.AboutDialog import AboutDialog
from DataFlow.SerialPortProcess import SerialPortProcess
from DataFlow.DataFlowProcess import DataFlowProcess
from Predict.PredictProcess import PredictProcess
from Control.DataHandleThread import DataHandleThread, ControlHandleThread
from Control.IOProcess import IOProcess


class MyWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()

        # TODO:后期log与一些缓存数据的存储路径
        self.save_root = utils.make_dir("temp")

        self.setupUi(self)

        # Action栏信号槽绑定
        self.Setting_action.triggered.connect(self.setting_action_trigger)
        self.Save_action.triggered.connect(self.save_action_trigger)
        self.Close_action.triggered.connect(self.close_action_trigger)
        self.About_action.triggered.connect(self.about_action_trigger)
        self.Login_action.triggered.connect(self.login_action_trigger)

        # Serial port相关按键信号槽绑定 and 串口发送edit栏正则化配置
        self.send_push_button.clicked.connect(self.send_data_to_serial_port)
        self.command_line_edit.setValidator(utils.regex_init('^[0-9a-fA-F]{1,2}$'))
        self.command_line_edit.textChanged.connect(lambda: utils.lowercase_to_uppercase(self.command_line_edit))
        self.data_line_edit.setValidator(utils.regex_init('^[0-9a-fA-F]{1,}$'))
        self.data_line_edit.textChanged.connect(lambda: utils.lowercase_to_uppercase(self.data_line_edit))

        # Result table属性设置
        # TODO:后期需要与神经网络输出结果对应
        self.predict_res_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.template_file_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # 进程参数
        # --进程间数据队列
        self.v2s_data_queue = Queue()
        self.s2v_data_queue = Queue()
        self.s2p_data_queue = Queue()
        self.d2p_data_queue = Queue()
        self.d2v_data_queue = Queue()

        # --进程间参数队列
        self.v2s_params_dict = Manager().dict()
        self.v2d_params_str = Manager().Value(c_char_p, "")

        # --进程间控制标志位
        self.v2s_control_flag = Value('i', const.SERIAL_PORT_IDLE)
        self.v2d_control_flag = Value('i', const.DATA_FLOW_IDLE)
        self.v2p_control_flag = Value('i', 0)

        # --进程间控制管道
        # V与S之间的管道
        self.vs_control_pipe, self.sv_control_pipe = Pipe(duplex=True)
        # V与D之间的管道
        self.vd_control_pipe, self.dv_control_pipe = Pipe(duplex=True)
        # V与P之间的管道
        # self.v_control_pipe, self.p2v_control_pipe = Pipe(duplex=True)

        # 串口进程初始化
        self.serial_port_process = SerialPortProcess(v2s_queue=self.v2s_data_queue,
                                                     s2v_queue=self.s2v_data_queue,
                                                     s2p_queue=self.s2p_data_queue,
                                                     v2s_params_dict=self.v2s_params_dict,
                                                     v2s_control_flag=self.v2s_control_flag,
                                                     sv_control_pipe=self.sv_control_pipe)
        self.serial_port_process.open()

        # 数据流进程初始化
        self.data_flow_process = DataFlowProcess(d2p_queue=self.d2p_data_queue,
                                                 d2v_queue=self.d2v_data_queue,
                                                 v2d_params_str=self.v2d_params_str,
                                                 v2d_control_flag=self.v2d_control_flag,
                                                 dv_control_pipe=self.dv_control_pipe)
        self.data_flow_process.open()

        # 预测进程初始化
        # self.predict_process = PredictProcess()

        # 截图进程初始化(后期废弃)
        self.io_process = IOProcess(s2p_queue=self.s2p_data_queue,
                                    d2p_queue=self.d2p_data_queue)
        self.io_process.open()

        # 界面处理数据队列线程初始化
        self.DataHandleThread = DataHandleThread(self)
        self.DataHandleThread.serial_port_receive_signal.connect(self.receive_data_from_serial_port)
        self.DataHandleThread.data_flow_receive_signal.connect(self.receive_data_from_data_flow)
        self.DataHandleThread.start()

        # 界面处理控制管道线程初始化
        self.ControlHandleThread = ControlHandleThread(self)
        self.ControlHandleThread.status_bar_info_signal.connect(self.status_show)
        self.ControlHandleThread.control_bar_visible_signal.connect(self.control_bar_visible_changed)
        self.ControlHandleThread.slider_init_signal.connect(self.slider_init)
        self.ControlHandleThread.slider_value_signal.connect(self.slider_value_changed)
        self.ControlHandleThread.current_frame_signal.connect(self.current_frame_changed)
        self.ControlHandleThread.total_frame_signal.connect(self.total_frame_changed)
        self.ControlHandleThread.play_button_triggered_signal.connect(self.button_icon_changed)
        self.ControlHandleThread.start()

        # 窗口相关参数
        self.play_bar_widget.setVisible(False)

        # Video相关按键信号槽绑定
        self.clicked_flag = True
        self.play_push_button.clicked.connect(self.video_play_button_trigger)
        self.faster_push_button.clicked.connect(self.video_faster_button_clicked)
        self.slower_push_button.clicked.connect(self.video_slower_button_clicked)
        self.video_progress_slider.sliderMoved.connect(self.video_slider_moved)

        # Training相关按键信号槽绑定
        self.train_push_Button.clicked.connect(lambda: self.model_running("Template.txt"))
        self.predict_push_button.clicked.connect(lambda: self.predict_push_button_clicked("spin001.txt"))
        self.compare_push_button.clicked.connect(self.compare_push_button_clicked)

    def input_source_changed(self, source):
        if self.v2d_params_str.value != source:
            self.v2d_params_str.value = source
            # source发生了切换，且之前的source为Video or Camera(USB\GigE) → 先进入Release再进入Change
            if self.v2d_control_flag.value == const.DATA_FLOW_RUN:
                self.v2d_control_flag.value = const.DATA_FLOW_RELEASE
            else:
                self.v2d_control_flag.value = const.DATA_FLOW_CHANGE

    '''Action bar signal slot callback function'''
    # 设置窗口触发
    def setting_action_trigger(self):
        setting_dialog = SettingDialog(self)
        setting_dialog.status_bar_signal.connect(self.status_show)
        result = setting_dialog.exec_()
        source = setting_dialog.get_cur_input_source()
        setting_dialog.destroy()
        if result:
            self.input_source_changed(source)

    # 保存窗口触发
    def save_action_trigger(self):
        root_dir = QFileDialog.getExistingDirectory(self, "Open Directory ", '//')
        if root_dir == "":
            self.save_root = os.getcwd()
        else:
            self.save_root = root_dir

    # 关闭主程序窗口触发
    def close_action_trigger(self):
        self.retreat_safely()
        Quit_App = QApplication.instance()
        Quit_App.quit()

    # 关于窗口触发
    @staticmethod
    def about_action_trigger():
        about_dialog = AboutDialog()
        _ = about_dialog.exec_()
        about_dialog.destroy()

    # 登录窗口触发(为后期预留的接口)
    def login_action_trigger(self):
        QMessageBox.information(self, 'Login window', "Login function will coming soon!", QMessageBox.Ok, QMessageBox.Ok)

    '''Data Flow interface signal slot callback function'''
    def status_show(self, show_text):
        self.statusbar.showMessage(show_text, msecs=5000)

    def slider_init(self, max_count):
        self.video_progress_slider.setMaximum(max_count)
        self.video_progress_slider.setMinimum(0)
        self.video_progress_slider.setSingleStep(1)

    def control_bar_visible_changed(self, visible):
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

    '''Model train/predict/compare signal slot callback function'''
    def model_running(self, sequence_save_file_name):
        if self.train_push_Button.isChecked():
            self.sequence_file_name.value = sequence_save_file_name
            self.predict_process_flag.value = const.PREDICT_PROCESS_STARTING
        else:
            self.predict_process_flag.value = const.PREDICT_PROCESS_STOPPING

    def predict_push_button_clicked(self, sequence_save_file_name):
        if self.predict_push_button.isChecked():
            self.sequence_file_name.value = sequence_save_file_name
            self.predict_process_flag.value = const.PREDICT_PROCESS_STARTING
        else:
            self.predict_process_flag.value = const.PREDICT_PROCESS_STOPPING

    # TODO: 后期需要加入选择模板的功能
    def compare_push_button_clicked(self):
        res, error_index, correct_rate = utils.compare('./temp/serializations', "spin001.txt", "Template.txt")
        cur_row = self.predict_res_tableWidget.rowCount() + 1
        self.predict_res_tableWidget.setRowCount(cur_row)
        newItem = QTableWidgetItem("{}%".format(correct_rate))
        self.predict_res_tableWidget.setItem(cur_row - 1, 0, newItem)

    '''Video signal slot callback function'''
    # 视频播放信号槽回调函数
    def video_play_button_trigger(self):
        if self.clicked_flag:
            pause_icon = QtGui.QIcon()
            pause_icon.addPixmap(QtGui.QPixmap(":/icon/pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.play_push_button.setIcon(pause_icon)
            self.clicked_flag = False
            self.vd_control_pipe.send(["V2D", const.PLAY_STATUS, True, None])
        else:
            play_icon = QtGui.QIcon()
            play_icon.addPixmap(QtGui.QPixmap(":/icon/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.play_push_button.setIcon(play_icon)
            self.clicked_flag = True
            self.vd_control_pipe.send(["V2D", const.PLAY_STATUS, False, None])

    # 视频快进信号槽回调函数
    def video_faster_button_clicked(self):
        self.vd_control_pipe.send(["V2D", const.FASTER_TRIGGER, True, None])

    # 视频快退信号槽回调函数
    def video_slower_button_clicked(self):
        self.vd_control_pipe.send(["V2D", const.SLOWER_TRIGGER, True, None])

    # 视频滑条信号槽回调函数
    def video_slider_moved(self):
        self.vd_control_pipe.send(["V2D", const.SLIDER_TRIGGER, self.video_progress_slider.value(), None])

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.retreat_safely()

    def retreat_safely(self):
        print("close")
        # TODO:兼顾进程安全推出,Camera安全退出
        if self.v2d_control_flag.value == const.DATA_FLOW_RUN:
            self.v2d_params_str.value = ""
            self.v2d_control_flag.value = const.DATA_FLOW_RELEASE
        time.sleep(0.1)
        self.serial_port_process.close()
        self.data_flow_process.close()
        # TODO:测试结束后删除
        self.io_process.close()

    def send_data_to_serial_port(self):
        try:
            if self.v2s_control_flag.value == const.SERIAL_PORT_RUN:
                command_length = len(self.command_line_edit.text())
                data_length = len(self.data_line_edit.text())
                if self.command_line_edit.text() and self.data_line_edit.text() and \
                        not command_length % 2 and not data_length % 2:
                    # 所有数据都是str形式
                    head = const.DATA_HEADER
                    length = utils.filling(utils.split_hex(hex((command_length + data_length) // 2 + 1)), 2)
                    data = self.command_line_edit.text() + self.data_line_edit.text()
                    total_data = head + length.upper() + data
                    # 发送bytes到串口进程
                    self.v2s_data_queue.put(bytes.fromhex(total_data))
                    pattern = re.compile('.{2}')
                    final_data = ' '.join(pattern.findall(total_data))
                    self.serial_port_data_text_browser.append(utils.colourful_text("[Send]-[{}]-[{}]".format(datetime.datetime.now().strftime('%H:%M:%S.%f'),
                                                                                                             final_data), "red"))
            else:
                QMessageBox.warning(self, 'Serial Port', "Open serial port before sending message", QMessageBox.Ok, QMessageBox.Ok)
        except Exception as error:
            print(error)

    # 数据处理API
    def receive_data_from_serial_port(self, data_list):
        pattern = re.compile('.{2}')
        final_data = ' '.join(pattern.findall(data_list[0]))
        self.serial_port_data_text_browser.append(utils.colourful_text("[Receive]-[{}]-[{}]-[{}]".format(datetime.datetime.now().strftime('%H:%M:%S.%f'),
                                                                                                         final_data.upper(), data_list[1]), "blue"))

    def receive_data_from_data_flow(self, img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            transform = const.BGR2RGB
        else:
            transform = const.GRAY2RGB
        utils.show_image_to_label(img, self.input_label, transform)
