# chat_app/chat_app.py
from PyQt6 import QtWidgets, QtCore
from .ui_setup import setup_ui

class ChatApp(QtWidgets.QWidget):
    COLORS = {
        "space_navy": "#ffffff",
        "steel_grey": "rgba(83, 92, 104, 0.2)",
        "white": "white",
        "han_blue": "#2F61D5",
        "golden_bay": "rgba(186, 220, 88,1.0)",
    }

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        setup_ui(self)

    def sendMessage(self):
        message = self.inputArea.messageInput.toPlainText().strip()
        if message:
            self.appendMessage(f"You: {message}", "sent")
            self.inputArea.messageInput.clear()
            QtCore.QTimer.singleShot(500, lambda: self.appendMessage(f"Bot: I am a mock-up reply for \"{message}\".", "received"))

    def appendMessage(self, message, messageType):
        messageContainer = QtWidgets.QLabel(message)
        
        style = (
            f"border-radius: 15px; "
            f"border: 1px solid {self.COLORS['steel_grey']}; "
            "padding: 10px;"
        )

        if messageType == "sent":
            style += "margin: 5px 10px 5px auto; font-size: 14px;"
        else:
            style += f"background-color: {self.COLORS['golden_bay']}; margin: 5px auto 5px 10px; font-size: 14px;"

        messageContainer.setStyleSheet(style)
        messageContainer.setWordWrap(True)

        # Add the message to the layout
        self.chatMessagesLayout.addWidget(messageContainer)
        self.animateMessageEntry(messageContainer)

        # Ensure the scroll area scrolls to the bottom
        self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())

    def animateMessageEntry(self, widget):
        # Simple fade-in animation
        fade_animation = QtCore.QPropertyAnimation(widget, b"windowOpacity")
        fade_animation.setDuration(500)  # Duration for fade-in effect
        fade_animation.setStartValue(0)
        fade_animation.setEndValue(1)
        fade_animation.start()

        # Store the reference to the animation to keep it from being garbage collected
        widget.animation = fade_animation