from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd

from RF_Iter_Missing import *
from DTClassifier import *

class Ui_MainWindow(QtWidgets.QMainWindow):

	def __init__(self):
		super(Ui_MainWindow,self).__init__()
		self.setupUi(self)
		self.retranslateUi(self)

	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(500, 500)
		self.centralWidget = QtWidgets.QWidget(MainWindow)
		self.centralWidget.setObjectName("centralWidget")
		self.retranslateUi(MainWindow)

		# load the train data
		self.loadTrainButton = QtWidgets.QPushButton(self.centralWidget)
		self.loadTrainButton.setGeometry(QtCore.QRect(190, 90, 75, 23))
		self.loadTrainButton.setObjectName("loadTrainButton")
		self.loadTrainButton.setText("loadTrain")
		self.loadTrainButton.clicked.connect(self.openTrainFile)

		# load the test data
		self.loadTestButton = QtWidgets.QPushButton(self.centralWidget)
		self.loadTestButton.setGeometry(QtCore.QRect(130, 100, 75, 23))
		self.loadTestButton.setObjectName("loadTestButton")
		self.loadTestButton.setText("LoadTest")
		self.loadTestButton.clicked.connect(self.openTestFile)

		# run the code
		self.runButton = QtWidgets.QPushButton(self.centralWidget)
		self.runButton.setGeometry(QtCore.QRect(150, 150, 75, 23))
		self.runButton.setObjectName("runButton")
		self.runButton.setText("Run")
		self.runButton.clicked.connect(self.run)

		# display the log information
		

		MainWindow.setCentralWidget(self.centralWidget)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)


	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "Dirty Data Model Generator"))


	def openTrainFile(self):
		openfile_name = QFileDialog.getOpenFileName(self,'选择文件','','Excel files(*.xlsx , *.xls， *csv)')
		self.fileName = openfile_name[0]
		self.data = pd.read_csv(self.fileName, header = 0, index_col = 0)
		print(self.data.head())
		print(self.fileName)

	def openTestFile(self):
		openfile_name = QFileDialog.getOpenFileName(self,'选择文件','','Excel files(*.xlsx , *.xls， *csv)')
		self.fileName = openfile_name[0]
		self.testdata = pd.read_csv(self.fileName, header = 0, index_col = 0)
		print(self.testdata.head())
		print(self.fileName)


	def run(self):
		warnings.filterwarnings(action='ignore', category=DeprecationWarning)
		model = DecisionTreeClassifier(criterion = 'entropy')
		test = RF_Iter_Missing(model, 0.99)
		new_data, p_all, p_fill = test.RF_Missing_Iterative(copy.deepcopy(self.data), copy.deepcopy(self.testdata), model, 3, 1)

		# bagging test
		ensemble_result = test.ensemble_model.bagging(self.testdata.iloc[:, :-1])
		accuracy = len(np.where(self.testdata.iloc[:, -1] == ensemble_result)[0])/self.testdata.shape[0]
		print("ensemble learning(bagging) accuracy over the dataset: {}".format(accuracy))

		#stacking test
		ensemble_result = test.ensemble_model.stacking(self.testdata.iloc[:, :-1])
		accuracy = len(np.where(self.testdata.iloc[:, -1] == ensemble_result)[0])/self.testdata.shape[0]
		print("ensemble learning(stacking) accuracy over the dataset: {}".format(accuracy))


if __name__ == "__main__":
	import sys
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	MainWindow.show()


	sys.exit(app.exec_())