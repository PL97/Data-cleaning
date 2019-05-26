# from PyQt5 import QtCore, QtGui, QtWidgets, Qt
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *
# import pandas as pd

# from RF_Iter_Missing import *
# from DTClassifier import *

# class Ui_MainWindow(QtWidgets.QMainWindow):

# 	def __init__(self):
# 		super(Ui_MainWindow,self).__init__()
# 		self.setupUi(self)
# 		self.retranslateUi(self)

# 	def setupUi(self, MainWindow):
# 		MainWindow.setObjectName("MainWindow")
# 		MainWindow.resize(500, 500)
# 		self.centralWidget = QtWidgets.QWidget(MainWindow)
# 		self.centralWidget.setObjectName("centralWidget")
# 		self.retranslateUi(MainWindow)

# 		# load the train data
# 		self.loadTrainButton = QtWidgets.QPushButton(self.centralWidget)
# 		self.loadTrainButton.setGeometry(QtCore.QRect(190, 90, 75, 23))
# 		self.loadTrainButton.setObjectName("loadTrainButton")
# 		self.loadTrainButton.setText("loadTrain")
# 		self.loadTrainButton.clicked.connect(self.openTrainFile)

# 		# load the test data
# 		self.loadTestButton = QtWidgets.QPushButton(self.centralWidget)
# 		self.loadTestButton.setGeometry(QtCore.QRect(130, 100, 75, 23))
# 		self.loadTestButton.setObjectName("loadTestButton")
# 		self.loadTestButton.setText("LoadTest")
# 		self.loadTestButton.clicked.connect(self.openTestFile)

# 		# run the code
# 		self.runButton = QtWidgets.QPushButton(self.centralWidget)
# 		self.runButton.setGeometry(QtCore.QRect(150, 150, 75, 23))
# 		self.runButton.setObjectName("runButton")
# 		self.runButton.setText("Run")
# 		self.runButton.clicked.connect(self.run)

# 		# display the log information
		

# 		MainWindow.setCentralWidget(self.centralWidget)
# 		QtCore.QMetaObject.connectSlotsByName(MainWindow)


# 	def retranslateUi(self, MainWindow):
# 		_translate = QtCore.QCoreApplication.translate
# 		MainWindow.setWindowTitle(_translate("MainWindow", "Dirty Data Model Generator"))


# 	def openTrainFile(self):
# 		openfile_name = QFileDialog.getOpenFileName(self,'选择文件','','Excel files(*.xlsx , *.xls， *csv)')
# 		self.fileName = openfile_name[0]
# 		self.data = pd.read_csv(self.fileName, header = 0, index_col = 0)
# 		print(self.data.head())
# 		print(self.fileName)

# 	def openTestFile(self):
# 		openfile_name = QFileDialog.getOpenFileName(self,'选择文件','','Excel files(*.xlsx , *.xls， *csv)')
# 		self.fileName = openfile_name[0]
# 		self.testdata = pd.read_csv(self.fileName, header = 0, index_col = 0)
# 		print(self.testdata.head())
# 		print(self.fileName)


# 	def run(self):
# 		warnings.filterwarnings(action='ignore', category=DeprecationWarning)
# 		model = DecisionTreeClassifier(criterion = 'entropy')
# 		test = RF_Iter_Missing(model, 0.99)
# 		new_data, p_all, p_fill = test.RF_Missing_Iterative(copy.deepcopy(self.data), copy.deepcopy(self.testdata), model, 3, 1)

# 		# bagging test
# 		ensemble_result = test.ensemble_model.bagging(self.testdata.iloc[:, :-1])
# 		accuracy = len(np.where(self.testdata.iloc[:, -1] == ensemble_result)[0])/self.testdata.shape[0]
# 		print("ensemble learning(bagging) accuracy over the dataset: {}".format(accuracy))

# 		#stacking test
# 		ensemble_result = test.ensemble_model.stacking(self.testdata.iloc[:, :-1])
# 		accuracy = len(np.where(self.testdata.iloc[:, -1] == ensemble_result)[0])/self.testdata.shape[0]
# 		print("ensemble learning(stacking) accuracy over the dataset: {}".format(accuracy))


# if __name__ == "__main__":
# 	import sys
# 	app = QtWidgets.QApplication(sys.argv)
# 	MainWindow = QtWidgets.QMainWindow()
# 	ui = Ui_MainWindow()
# 	ui.setupUi(MainWindow)
# 	MainWindow.show()


# 	sys.exit(app.exec_())

import sys
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon
import pandas as pd
from RF_Iter_Missing import *
from DTClassifier import *

class EmittingStream(QtCore.QObject):  
		textWritten = QtCore.pyqtSignal(str)  #定义一个发送str的信号
		def write(self, text):
			self.textWritten.emit(str(text))

class RunThread(QThread):#线程类

	def __init__(self, data, testdata):
		super(RunThread, self).__init__()
		self.data = data
		self.testdata = testdata

	def __del__(self):
		self.wait()


	def run(self): #线程执行函数
		print("training for the ensemble model...")
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

class LoadDataThread(QThread):

	def __init__(self, input_table, tableWidget):
		super(LoadDataThread, self).__init__()
		self.input_table = input_table
		self.tableWidget = tableWidget

	def __del__(self):
		self.wait()

	def run(self):
		if self.input_table.shape[0] > 0 and self.input_table.shape[1] > 0:
			input_table_header = self.input_table.columns.values.tolist()

			# set some basic configure information for the display table
			self.tableWidget.setColumnCount(self.input_table.shape[1])
			self.tableWidget.setRowCount(self.input_table.shape[0])
			self.tableWidget.setHorizontalHeaderLabels(input_table_header)

			# put all the value to the table(by row)
			for index, row in self.input_table.iterrows():
				for j in range(len(row)):
					input_table_items = str(row[j])
					newItem = QTableWidgetItem(input_table_items) 
					newItem.setTextAlignment(Qt.AlignHCenter|Qt.AlignVCenter)
					self.tableWidget.setItem(index, j, newItem)
		else:
			# self.centralWidget.show()
			pass

class LoadData(QtCore.QThread):
 
	def __init__(self, gui, mode):
		super(Runthread, self).__init__()
		self.gui = gui
 
	def __del__(self):
		self.wait()
 
	def run(self):
		gui.openFile(self.mode)

class Example(QtWidgets.QMainWindow):
	
	def __init__(self):
		super().__init__()
		
		self.initUI()
		
		
	def initUI(self):
		
		# menu for adding train file
		addFile = QtWidgets.QAction(QIcon('add.jpg'), 'add train file', self)
		addFile.setShortcut('Ctrl+A')
		addFile.triggered.connect(lambda: self.openFile(1))
		addFile.setStatusTip('add train file')

		# menu for adding test file
		addTest = QtWidgets.QAction(QIcon('add.jpg'), 'add test file', self)
		addTest.setShortcut('Ctrl+A+T')
		addTest.triggered.connect(lambda: self.openFile(1))
		addTest.setStatusTip('add test file')
		
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(addFile)
		fileMenu.addAction(addTest)

		# display the infromation about the input train data
		self.rclb = QLabel(self)
		self.rclb.setText("row:")
		self.rclb.setGeometry(QtCore.QRect(510, 70, 80, 20))
		self.rcle = QLabel(self)
		self.rcle.setText("None")
		self.rcle.setGeometry(QtCore.QRect(600, 70, 150, 20))
		self.rcle.setObjectName("row count")

		self.mclb = QLabel(self)
		self.mclb.setText("missing:")
		self.mclb.setGeometry(QtCore.QRect(510, 100, 80, 20))
		self.mcle = QLabel(self)
		self.mcle.setText("None")
		self.mcle.setGeometry(QtCore.QRect(600, 100, 150, 20))
		self.mcle.setObjectName("missing count")

		self.mrlb = QLabel(self)
		self.mrlb.setText("missing rate:")
		self.mrlb.setGeometry(QtCore.QRect(510, 130, 80, 20))
		self.mrle = QLabel(self)
		self.mrle.setText("None")
		self.mrle.setGeometry(QtCore.QRect(600, 130, 150, 20))
		self.mrle.setObjectName("lineEdit")


		# display the input train data
		self.tableWidget = QtWidgets.QTableWidget(self)
		self.tableWidget.setGeometry(QtCore.QRect(0, 60, 500, 200))
		self.tableWidget.setObjectName("tableWidget")
		self.tableWidget.setColumnCount(0)
		self.tableWidget.setRowCount(0)
		self.tableWidget.setStyleSheet("selection-background-color:pink")
		self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
		self.tableWidget.raise_()

		# display the log infromation
		self.textEdit = QtWidgets.QTextEdit(self)
		self.textEdit.setGeometry(QtCore.QRect(0, 200, 500, 200))
		self.textEdit.setObjectName("textEdit")
		# self.textEdit.setReadOnly(True)
		sys.stdout = EmittingStream(textWritten=self.outputWritten)  
		sys.stderr = EmittingStream(textWritten=self.outputWritten) 

		# run the code
		self.runButton = QtWidgets.QPushButton(self)
		self.runButton.setGeometry(QtCore.QRect(600, 200, 75, 23))
		self.runButton.setObjectName("runButton")
		self.runButton.setText("Run")
		self.runButton.clicked.connect(self.clickRun)

		self.resize(700, 500)
		self.center()
		self.setWindowTitle('Dirty Data Model Generator')
		self.show()

	# add context menu
	def contextMenuEvent(self, event):
		cmenu = QMenu(self)
		newTrain = cmenu.addAction("Add Train File")
		newTest = cmenu.addAction("Add Test File")
		action = cmenu.exec_(self.mapToGlobal(event.pos()))
		if action == newTrain:
			self.openFile(1)
		elif action == newTest:
			self.openFile(0)
		else:
			pass
		
	def center(self):
		
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def openFile(self, mode):
		openfile_name = QFileDialog.getOpenFileName(self,'选择文件','','Excel files(*.xlsx , *.xls， *csv)')
		if mode == 1:
			self.trainfileName = openfile_name[0]
			self.data = pd.read_csv(self.trainfileName, header = 0, index_col = 0)
			self.rcle.setText(str(self.data.shape[0]))
			complete_count = self.data.shape[0] * self.data.shape[1]
			missing_count = complete_count - sum(self.data.count())
			self.mcle.setText(str(missing_count))
			self.mrle.setText(str(missing_count/complete_count))
			self.loaddatathread = LoadDataThread(self.data, self.tableWidget)
			self.loaddatathread.start()
			print("load train file successfully!")
		else:
			self.testfileName = openfile_name[0]
			self.testdata = pd.read_csv(self.testfileName, header = 0, index_col = 0)
			print("load test file successfully!")
		
	def openFile1(self, mode):
		openfile_name = QFileDialog.getOpenFileName(self,'选择文件','','Excel files(*.xlsx , *.xls， *csv)')
		if mode == 1:
			self.trainfileName = openfile_name[0]
			self.data = pd.read_csv(self.trainfileName, header = 0, index_col = 0)
			self.rcle.setText(str(self.data.shape[0]))
			complete_count = self.data.shape[0] * self.data.shape[1]
			missing_count = complete_count - sum(self.data.count())
			self.mcle.setText(str(missing_count))
			self.mrle.setText(str(missing_count/complete_count))
			# self.creat_table_show(self.data, self.tableWidget)
			print("load train file successfully!")
		else:
			self.testfileName = openfile_name[0]
			self.testdata = pd.read_csv(self.testfileName, header = 0, index_col = 0)
			print("load test file successfully!")

	def creat_table_show(self, input_table, tableWidget):
		###===========读取表格，转换表格，===========================================
		if input_table.shape[0] > 0 and input_table.shape[1] > 0:
			input_table_rows = input_table.shape[0]
			input_table_colunms = input_table.shape[1]
			input_table_header = input_table.columns.values.tolist()

			# set some basic configure information for the display table
			tableWidget.setColumnCount(input_table_colunms)
			tableWidget.setRowCount(input_table_rows)
			tableWidget.setHorizontalHeaderLabels(input_table_header)

			# put all the value to the table(by row)
			for index, row in input_table.iterrows():
				for j in range(len(row)):
					if index > 100:
						break
					input_table_items = str(row[j])
					newItem = QTableWidgetItem(input_table_items) 
					newItem.setTextAlignment(Qt.AlignHCenter|Qt.AlignVCenter)
					tableWidget.setItem(index, j, newItem)
		else:
			self.centralWidget.show()


	def outputWritten(self, text):
		cursor = self.textEdit.textCursor()
		cursor.movePosition(QtGui.QTextCursor.End)
		cursor.insertText(text)
		self.textEdit.setTextCursor(cursor)
		self.textEdit.ensureCursorVisible()

 
	def call_backlog(self, msg):
		self.pbar.setValue(int(msg))  # 将线程的参数传入进度条

	def clickRun(self):
		self.my_thread = RunThread(self.data, self.testdata)
		self.my_thread.start()



if __name__ == '__main__':
	
	app = QApplication(sys.argv)
	ex = Example()
	sys.exit(app.exec_())