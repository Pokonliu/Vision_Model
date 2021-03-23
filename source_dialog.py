from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui_setting_dialog import Ui_dialog
from SearchDeviceThread import SearchDeviceThread


class SourceDialog(QDialog, Ui_dialog):
    status_bar_signal = pyqtSignal(str)

    def __init__(self, UI):
        super().__init__()
        self.setupUi(self)

        self.search_device_thread = SearchDeviceThread(self)
        self.super_UI = UI

        self.image_radio_button.toggled.connect(self.image_radio_button_toggled)
        self.image_tool_button.clicked.connect(self.image_tool_button_clicked)

        self.video_radio_button.toggled.connect(self.video_radio_button_toggled)
        self.video_tool_button.clicked.connect(self.video_tool_button_clicked)

        self.camera_radio_button.toggled.connect(self.camera_radio_button_toggled)
        self.camera_combo_box.currentIndexChanged.connect(self.camera_combo_box_current_index_changed)

        self.search_push_button.clicked.connect(self.search_push_button_clicked)
        self.search_device_thread.finished.connect(self.finish_show)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def finish_show(self):
        self.status_bar_signal.emit("New devices has been discovered")

    def search_push_button_clicked(self):
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
        self.label_output.setText(file_name)

    def video_tool_button_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', '//', "Image files (*.avi *.mp4 *.wmv *.rmvb)",)
        self.label_output.setText(file_name)

    def camera_combo_box_current_index_changed(self):
        self.label_output.setText(self.camera_combo_box.currentText())

    def get_label_result(self):
        return self.label_output.text()
