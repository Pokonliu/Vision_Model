import sys
from multiprocessing import freeze_support
from PyQt5.QtWidgets import *

from Control.Gunder_Multithread import MyWindow


if __name__ == '__main__':
    freeze_support()
    app = QApplication(sys.argv)
    ex = MyWindow()
    ex.show()
    sys.exit(app.exec_())
