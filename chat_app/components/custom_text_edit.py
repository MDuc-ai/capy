
from PyQt6 import QtWidgets, QtCore
class CustomTextEdit(QtWidgets.QTextEdit):
    enterPressed = QtCore.pyqtSignal()  # Signal for enter key

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Return:
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier:
                # If shift is pressed with Enter, insert a newline
                self.insertPlainText("\n")
            else:
                # Emit a signal when Enter is pressed without Shift
                self.enterPressed.emit()
        else:
            # Call the base class method for other keys
            super().keyPressEvent(event)
