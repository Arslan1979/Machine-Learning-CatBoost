import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from PyQt5.uic import loadUi
from pickle import load

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('design.ui', self)
        self.prediksiButton.clicked.connect(self.Btnprediksi)

    @pyqtSlot()
    def Btnprediksi(self):
        Tx = float(self.editTx.text())
        rr = float(self.editRR.text())

        if self.radioYa.isChecked():
            hh = 1
        elif self.radioTidak.isChecked():
            hh = 0

        self.labelHasil.setText("Predicting...")

        row = [[Tx, rr, hh]]

        result = self.predict(row)

        if result == 0:
            self.labelHasil.setText("Besok Tidak Hujan")
        elif result == 1:
            self.labelHasil.setText("Besok Hujan")

    def predict(self, row):
        print("Predicting...")
        model = load(open('model2.pkl', 'rb'))
        yhat = model.predict(row)

        return yhat[0]

app = QtWidgets.QApplication(sys.argv)

window = ShowImage()
window.setWindowTitle('Applikasi Prediksi Besok Hujan')
window.show()
sys.exit(app.exec_())
