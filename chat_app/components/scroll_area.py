from PyQt6 import QtWidgets, QtCore

class ScrollArea(QtWidgets.QScrollArea):
    def __init__(self, chatMessages):
        super().__init__()
        self.setWidgetResizable(True)
        self.setWidget(chatMessages)
        self.setStyleSheet("""
            QScrollArea {
                border: none; 
                background: transparent; 
            }
            QScrollBar {
                background: transparent; 
                width: 0px; 
                height: 0px; 
            }
        """)