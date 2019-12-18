from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap
from face_verification import verification_gui

import sys
import os


def pressed_button():
    # Getting paths to the photos
    path1 = win.lineEdit.text()
    path2 = win.lineEdit_2.text()

    # Check if the files exist
    exists = os.path.isfile(path1) and os.path.isfile(path2)

    if exists:
        pixmap_1 = QPixmap(path1)
        win.label.setPixmap(pixmap_1)
        pixmap_2 = QPixmap(path2)
        win.label_2.setPixmap(pixmap_2)
        win.label_3.setText("Processing...")
        win.label_3.setStyleSheet("QLabel#label_3 {color: black}")
        app.processEvents()

        # Verification part
        match, score = verification_gui(path1, path2)

        if match:  # the same person
            win.label_3.setText("It is the same person!  Cosine distance = %f" % score)
            win.label_3.setStyleSheet("QLabel#label_3 {color: green}")

        else:  # different person
            win.label_3.setText("It is different person!  Cosine distance = %f" % score)
            win.label_3.setStyleSheet("QLabel#label_3 {color: red}")

    else:
        # If files don't exist, show error
        win.label_3.setText("One of the files (or both) do not exist.")


app = QtWidgets.QApplication([])
win = uic.loadUi("gui.ui")  # location of template
win.pushButton.clicked.connect(pressed_button)
win.show()
sys.exit(app.exec())
