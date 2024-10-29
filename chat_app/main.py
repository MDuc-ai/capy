# main.py
from PyQt6 import QtWidgets
import sys
from components.chat_app import ChatApp

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    chatApp = ChatApp()
    chatApp.show()
    sys.exit(app.exec())