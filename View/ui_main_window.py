# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_bar_detection.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(906, 719)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/mainwindow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_input = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_input.setFont(font)
        self.groupBox_input.setObjectName("groupBox_input")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_input)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.input_label = QtWidgets.QLabel(self.groupBox_input)
        self.input_label.setText("")
        self.input_label.setScaledContents(True)
        self.input_label.setAlignment(QtCore.Qt.AlignCenter)
        self.input_label.setObjectName("input_label")
        self.verticalLayout_4.addWidget(self.input_label)
        self.play_bar_widget = QtWidgets.QWidget(self.groupBox_input)
        self.play_bar_widget.setObjectName("play_bar_widget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.play_bar_widget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.slower_push_button = QtWidgets.QPushButton(self.play_bar_widget)
        self.slower_push_button.setMinimumSize(QtCore.QSize(30, 30))
        self.slower_push_button.setMaximumSize(QtCore.QSize(30, 30))
        self.slower_push_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icon/slower.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.slower_push_button.setIcon(icon1)
        self.slower_push_button.setIconSize(QtCore.QSize(30, 30))
        self.slower_push_button.setFlat(True)
        self.slower_push_button.setObjectName("slower_push_button")
        self.horizontalLayout_3.addWidget(self.slower_push_button)
        self.play_push_button = QtWidgets.QPushButton(self.play_bar_widget)
        self.play_push_button.setMinimumSize(QtCore.QSize(30, 30))
        self.play_push_button.setMaximumSize(QtCore.QSize(30, 30))
        self.play_push_button.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icon/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.play_push_button.setIcon(icon2)
        self.play_push_button.setIconSize(QtCore.QSize(30, 30))
        self.play_push_button.setFlat(True)
        self.play_push_button.setObjectName("play_push_button")
        self.horizontalLayout_3.addWidget(self.play_push_button)
        self.faster_push_button = QtWidgets.QPushButton(self.play_bar_widget)
        self.faster_push_button.setMinimumSize(QtCore.QSize(30, 30))
        self.faster_push_button.setMaximumSize(QtCore.QSize(30, 30))
        self.faster_push_button.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icon/faster.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.faster_push_button.setIcon(icon3)
        self.faster_push_button.setIconSize(QtCore.QSize(30, 30))
        self.faster_push_button.setFlat(True)
        self.faster_push_button.setObjectName("faster_push_button")
        self.horizontalLayout_3.addWidget(self.faster_push_button)
        self.video_progress_slider = QtWidgets.QSlider(self.play_bar_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_progress_slider.sizePolicy().hasHeightForWidth())
        self.video_progress_slider.setSizePolicy(sizePolicy)
        self.video_progress_slider.setStyleSheet("QSlider::groove:horizontal {\n"
"border: 1px solid #bbb;\n"
"background: white;\n"
"height: 10px;\n"
"border-radius: 4px;\n"
"}\n"
" \n"
"QSlider::sub-page:horizontal {\n"
"background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,\n"
"    stop: 0 #66e, stop: 1 #bbf);\n"
"background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,\n"
"    stop: 0 #bbf, stop: 1 #55f);\n"
"border: 1px solid #777;\n"
"height: 10px;\n"
"border-radius: 4px;\n"
"}\n"
" \n"
"QSlider::add-page:horizontal {\n"
"background: #fff;\n"
"border: 1px solid #777;\n"
"height: 10px;\n"
"border-radius: 4px;\n"
"}\n"
" \n"
"QSlider::handle:horizontal {\n"
"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
"    stop:0 #eee, stop:1 #ccc);\n"
"border: 1px solid #777;\n"
"width: 13px;\n"
"margin-top: -2px;\n"
"margin-bottom: -2px;\n"
"border-radius: 4px;\n"
"}\n"
" \n"
"QSlider::handle:horizontal:hover {\n"
"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
"    stop:0 #fff, stop:1 #ddd);\n"
"border: 1px solid #444;\n"
"border-radius: 4px;\n"
"}\n"
" \n"
"QSlider::sub-page:horizontal:disabled {\n"
"background: #bbb;\n"
"border-color: #999;\n"
"}\n"
" \n"
"QSlider::add-page:horizontal:disabled {\n"
"background: #eee;\n"
"border-color: #999;\n"
"}\n"
" \n"
"QSlider::handle:horizontal:disabled {\n"
"background: #eee;\n"
"border: 1px solid #aaa;\n"
"border-radius: 4px;\n"
"}\n"
"")
        self.video_progress_slider.setOrientation(QtCore.Qt.Horizontal)
        self.video_progress_slider.setObjectName("video_progress_slider")
        self.horizontalLayout_3.addWidget(self.video_progress_slider)
        self.cur_frame_label = QtWidgets.QLabel(self.play_bar_widget)
        self.cur_frame_label.setMinimumSize(QtCore.QSize(60, 0))
        self.cur_frame_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.cur_frame_label.setObjectName("cur_frame_label")
        self.horizontalLayout_3.addWidget(self.cur_frame_label)
        self.separae_label = QtWidgets.QLabel(self.play_bar_widget)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.separae_label.setFont(font)
        self.separae_label.setObjectName("separae_label")
        self.horizontalLayout_3.addWidget(self.separae_label)
        self.total_frame_label = QtWidgets.QLabel(self.play_bar_widget)
        self.total_frame_label.setMinimumSize(QtCore.QSize(60, 0))
        self.total_frame_label.setObjectName("total_frame_label")
        self.horizontalLayout_3.addWidget(self.total_frame_label)
        self.verticalLayout_4.addWidget(self.play_bar_widget)
        self.verticalLayout_4.setStretch(0, 100)
        self.verticalLayout_4.setStretch(1, 1)
        self.horizontalLayout.addWidget(self.groupBox_input)
        self.groupBox_output = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_output.setFont(font)
        self.groupBox_output.setObjectName("groupBox_output")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_output)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.output_label = QtWidgets.QLabel(self.groupBox_output)
        self.output_label.setText("")
        self.output_label.setAlignment(QtCore.Qt.AlignCenter)
        self.output_label.setObjectName("output_label")
        self.horizontalLayout_4.addWidget(self.output_label)
        self.horizontalLayout.addWidget(self.groupBox_output)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox_result = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_result.setMinimumSize(QtCore.QSize(0, 200))
        self.groupBox_result.setMaximumSize(QtCore.QSize(16777215, 200))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_result.setFont(font)
        self.groupBox_result.setObjectName("groupBox_result")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.groupBox_result)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.tabWidget = QtWidgets.QTabWidget(self.groupBox_result)
        self.tabWidget.setObjectName("tabWidget")
        self.predict_res_tab = QtWidgets.QWidget()
        self.predict_res_tab.setObjectName("predict_res_tab")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.predict_res_tab)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.predict_res_tableWidget = QtWidgets.QTableWidget(self.predict_res_tab)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.predict_res_tableWidget.setFont(font)
        self.predict_res_tableWidget.setObjectName("predict_res_tableWidget")
        self.predict_res_tableWidget.setColumnCount(5)
        self.predict_res_tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.predict_res_tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.predict_res_tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.predict_res_tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.predict_res_tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.predict_res_tableWidget.setHorizontalHeaderItem(4, item)
        self.predict_res_tableWidget.horizontalHeader().setDefaultSectionSize(120)
        self.verticalLayout_5.addWidget(self.predict_res_tableWidget)
        self.tabWidget.addTab(self.predict_res_tab, "")
        self.template_file_tab = QtWidgets.QWidget()
        self.template_file_tab.setObjectName("template_file_tab")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.template_file_tab)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.template_file_tableWidget = QtWidgets.QTableWidget(self.template_file_tab)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.template_file_tableWidget.setFont(font)
        self.template_file_tableWidget.setObjectName("template_file_tableWidget")
        self.template_file_tableWidget.setColumnCount(2)
        self.template_file_tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.template_file_tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.template_file_tableWidget.setHorizontalHeaderItem(1, item)
        self.template_file_tableWidget.horizontalHeader().setDefaultSectionSize(200)
        self.verticalLayout_3.addWidget(self.template_file_tableWidget)
        self.tabWidget.addTab(self.template_file_tab, "")
        self.serial_port_tab = QtWidgets.QWidget()
        self.serial_port_tab.setObjectName("serial_port_tab")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.serial_port_tab)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.serial_port_data_text_browser = QtWidgets.QTextBrowser(self.serial_port_tab)
        self.serial_port_data_text_browser.setObjectName("serial_port_data_text_browser")
        self.horizontalLayout_5.addWidget(self.serial_port_data_text_browser)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.command_label = QtWidgets.QLabel(self.serial_port_tab)
        self.command_label.setObjectName("command_label")
        self.verticalLayout_6.addWidget(self.command_label)
        self.command_line_edit = QtWidgets.QLineEdit(self.serial_port_tab)
        self.command_line_edit.setObjectName("command_line_edit")
        self.verticalLayout_6.addWidget(self.command_line_edit)
        self.data_label = QtWidgets.QLabel(self.serial_port_tab)
        self.data_label.setObjectName("data_label")
        self.verticalLayout_6.addWidget(self.data_label)
        self.data_line_edit = QtWidgets.QLineEdit(self.serial_port_tab)
        self.data_line_edit.setObjectName("data_line_edit")
        self.verticalLayout_6.addWidget(self.data_line_edit)
        self.send_push_button = QtWidgets.QPushButton(self.serial_port_tab)
        self.send_push_button.setObjectName("send_push_button")
        self.verticalLayout_6.addWidget(self.send_push_button)
        self.horizontalLayout_5.addLayout(self.verticalLayout_6)
        self.horizontalLayout_5.setStretch(0, 4)
        self.horizontalLayout_5.setStretch(1, 1)
        self.tabWidget.addTab(self.serial_port_tab, "")
        self.horizontalLayout_7.addWidget(self.tabWidget)
        self.horizontalLayout_2.addWidget(self.groupBox_result)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.train_push_Button = QtWidgets.QPushButton(self.centralwidget)
        self.train_push_Button.setMinimumSize(QtCore.QSize(50, 50))
        self.train_push_Button.setMaximumSize(QtCore.QSize(50, 50))
        self.train_push_Button.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icon/template.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.train_push_Button.setIcon(icon4)
        self.train_push_Button.setIconSize(QtCore.QSize(50, 50))
        self.train_push_Button.setCheckable(True)
        self.train_push_Button.setFlat(True)
        self.train_push_Button.setObjectName("train_push_Button")
        self.verticalLayout.addWidget(self.train_push_Button, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.predict_push_button = QtWidgets.QPushButton(self.centralwidget)
        self.predict_push_button.setMinimumSize(QtCore.QSize(50, 50))
        self.predict_push_button.setMaximumSize(QtCore.QSize(50, 50))
        self.predict_push_button.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icon/predict.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.predict_push_button.setIcon(icon5)
        self.predict_push_button.setIconSize(QtCore.QSize(50, 50))
        self.predict_push_button.setCheckable(True)
        self.predict_push_button.setFlat(True)
        self.predict_push_button.setObjectName("predict_push_button")
        self.verticalLayout.addWidget(self.predict_push_button, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.compare_push_button = QtWidgets.QPushButton(self.centralwidget)
        self.compare_push_button.setMinimumSize(QtCore.QSize(50, 50))
        self.compare_push_button.setMaximumSize(QtCore.QSize(50, 50))
        self.compare_push_button.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icon/compare.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.compare_push_button.setIcon(icon6)
        self.compare_push_button.setIconSize(QtCore.QSize(50, 50))
        self.compare_push_button.setFlat(True)
        self.compare_push_button.setObjectName("compare_push_button")
        self.verticalLayout.addWidget(self.compare_push_button, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 906, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.statusbar.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.statusbar.setFont(font)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.Setting_action = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icon/setting.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Setting_action.setIcon(icon7)
        self.Setting_action.setObjectName("Setting_action")
        self.Save_action = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/icon/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Save_action.setIcon(icon8)
        self.Save_action.setObjectName("Save_action")
        self.Close_action = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/icon/close.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Close_action.setIcon(icon9)
        self.Close_action.setObjectName("Close_action")
        self.About_action = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/icon/about.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.About_action.setIcon(icon10)
        self.About_action.setObjectName("About_action")
        self.Login_action = QtWidgets.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(":/icon/log.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Login_action.setIcon(icon11)
        self.Login_action.setObjectName("Login_action")
        self.toolBar.addAction(self.Login_action)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.Setting_action)
        self.toolBar.addAction(self.Save_action)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.About_action)
        self.toolBar.addAction(self.Close_action)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Gunder"))
        self.groupBox_input.setTitle(_translate("MainWindow", "Input"))
        self.slower_push_button.setToolTip(_translate("MainWindow", "快退"))
        self.play_push_button.setToolTip(_translate("MainWindow", "播放/暂停"))
        self.faster_push_button.setToolTip(_translate("MainWindow", "快进"))
        self.cur_frame_label.setText(_translate("MainWindow", "000"))
        self.separae_label.setText(_translate("MainWindow", ":"))
        self.total_frame_label.setToolTip(_translate("MainWindow", "总帧"))
        self.total_frame_label.setText(_translate("MainWindow", "000"))
        self.groupBox_output.setTitle(_translate("MainWindow", "Output"))
        self.groupBox_result.setTitle(_translate("MainWindow", "Result"))
        item = self.predict_res_tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Correct rate"))
        item = self.predict_res_tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Frame count"))
        item = self.predict_res_tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Needle count"))
        item = self.predict_res_tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Template label"))
        item = self.predict_res_tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Predict label"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.predict_res_tab), _translate("MainWindow", "Predict result"))
        item = self.template_file_tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "File path"))
        item = self.template_file_tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Needle total counts"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.template_file_tab), _translate("MainWindow", "Template file"))
        self.command_label.setText(_translate("MainWindow", "Commond"))
        self.data_label.setText(_translate("MainWindow", "Data"))
        self.send_push_button.setText(_translate("MainWindow", "Send"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.serial_port_tab), _translate("MainWindow", "Serial Port"))
        self.train_push_Button.setToolTip(_translate("MainWindow", "制作模板"))
        self.predict_push_button.setToolTip(_translate("MainWindow", "预测"))
        self.compare_push_button.setToolTip(_translate("MainWindow", "比较"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.Setting_action.setText(_translate("MainWindow", "输入源"))
        self.Setting_action.setToolTip(_translate("MainWindow", "输入源"))
        self.Save_action.setText(_translate("MainWindow", "保存"))
        self.Save_action.setToolTip(_translate("MainWindow", "保存"))
        self.Close_action.setText(_translate("MainWindow", "关闭"))
        self.Close_action.setToolTip(_translate("MainWindow", "关闭"))
        self.About_action.setText(_translate("MainWindow", "关于"))
        self.About_action.setToolTip(_translate("MainWindow", "关于"))
        self.Login_action.setText(_translate("MainWindow", "登录"))
        self.Login_action.setToolTip(_translate("MainWindow", "登录"))
import Source.source
