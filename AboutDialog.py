from PyQt5.QtWidgets import *
from ui_about import Ui_Dialog


class AboutDialog(QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
