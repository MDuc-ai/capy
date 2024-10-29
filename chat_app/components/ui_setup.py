from PyQt6 import QtWidgets, QtCore
from .scroll_area import ScrollArea
from .input_area import InputArea

def setup_ui(chat_app):
    chat_app.setWindowTitle("Chat App")
    chat_app.setGeometry(300, 300, 400, 600)
    chat_app.setStyleSheet(f"background-color: {chat_app.COLORS['space_navy']};")

    layout = QtWidgets.QVBoxLayout(chat_app)

    # Chat messages area
    chat_app.chatMessages = QtWidgets.QWidget()
    chat_app.chatMessagesLayout = QtWidgets.QVBoxLayout(chat_app.chatMessages)
    chat_app.chatMessagesLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
    chat_app.chatMessages.setStyleSheet("background: transparent;")

    chat_app.scrollArea = ScrollArea(chat_app.chatMessages)

    # Input area
    chat_app.inputArea = InputArea(chat_app.sendMessage, chat_app.COLORS)

    layout.addWidget(chat_app.scrollArea)
    layout.addWidget(chat_app.inputArea)