# chat_app/input_area.py
from PyQt6 import QtWidgets, QtCore, QtGui
from .custom_text_edit import CustomTextEdit

class InputArea(QtWidgets.QWidget):
    def __init__(self, sendMessageCallback, colors):
        super().__init__()
        self.sendMessageCallback = sendMessageCallback
        self.colors = colors
        self.initUI()

    def initUI(self):
        self.setFixedHeight(80)  # Set initial height to 80 to accommodate the input area and button
        layout = QtWidgets.QHBoxLayout(self)
        self.setStyleSheet(f"""
            QWidget {{
                border: 2px solid {self.colors['steel_grey']};
                border-radius: 10px;
                background-color: {self.colors['space_navy']};
            }}
        """)

        # Message input
        self.messageInput = CustomTextEdit()
        self.messageInput.enterPressed.connect(self.sendMessage)
        self.messageInput.setPlaceholderText("Type a message...")
        self.messageInput.setStyleSheet(f"""
            padding: 10px;
            border: none;
            background-color: {self.colors['white']}; 
            font-size: 14px;
            font-family: Arial, sans-serif;
        """)
        self.messageInput.setFixedHeight(50)  # Set initial height to 50

        # Send button
        self.sendButton = QtWidgets.QPushButton()
        self.sendButton.setIcon(QtGui.QIcon("icons/send.png"))  # Ensure icon path is correct
        self.sendButton.setIconSize(QtCore.QSize(24, 24))
        self.sendButton.setStyleSheet(f"""
            QPushButton {{
                border-radius: 10px; 
                padding: 10px;
                color: white;
            }}
        """)
        self.sendButton.setFixedHeight(50)  # Matched height
        
        self.sendButton.clicked.connect(self.sendMessage)

        layout.addWidget(self.messageInput)
        layout.addWidget(self.sendButton)

    def sendMessage(self):
        self.sendMessageCallback()