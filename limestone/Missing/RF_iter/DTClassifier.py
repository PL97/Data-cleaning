from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
from sklearn import preprocessing

def readFile(fname):
	return pd.read_csv(fname, header = 0)

def read_from_txt(fname):
	data = np.loadtxt(fname, delimiter = '\s', dtype = 'str')
	temp_data = []
	for x in data:
		temp_data.append(re.split(r'\s+', x))
	temp_data = np.array(temp_data)
	test_txt_or_num = temp_data[0]
	is_num = [i.replace(".", '', 1).isdigit() for i in test_txt_or_num]

	new_data = pd.DataFrame(temp_data)
	for i in range(len(is_num)):
		if is_num[i]:
			new_data[i] = new_data[i].astype(float)
	return new_data

def deal_with_catogory(data):
	num = preprocessing.LabelEncoder()
	num2Cat = {}
	for x in data.columns:
		if data[x].dtype != 'float':
			data[x] = num.fit_transform(data[x])
			num2Cat[x] = {i:num.classes_[i] for i  in range(len(num.classes_))}
	print(num2Cat)
	return data

# def DTclassifier(data)

if __name__ == '__main__':
	# l = ['0.58', '0.61', '0.47', '0.13', '0.5', '0.0', '0.48', '0.22']
	# l_t = list(map(float, l))
	# print(l_t)
	# afd
	fname = 'yeast/yeast.txt'
	data = read_from_txt(fname).loc[:, 1:]
	data = deal_with_catogory(data)

	features = data.columns
	X = data.drop(features[-1], axis = 1)
	y = data.loc[:, features[-1]]
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
	model = DecisionTreeClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	accuracy = len(np.where(y_test == y_pred)[0])/len(y_test)
	print(accuracy)

