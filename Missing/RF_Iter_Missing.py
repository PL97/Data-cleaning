import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import warnings
import random


class RF_Iter_Missing(object):

	def __init__(self):
		pass

	def Preprocessing_Iterative(self, data):
		# # prepare for the data
		# data = pd.read_csv('missing0.3.csv', header = 0, index_col = 0)
		missing = [x for x in data.columns if data[x].isnull().any()]
		# the index of tuple with some missing features
		missing_lines = {x:np.where(data[x].isnull())[0] for x in missing}
		xx, yy = np.where(data.isnull())
		l = len(xx)
		# ground_truth = data.iloc(xx[1])
		ground_data = pd.read_csv('iris.csv', header = 0)
		ground_truth = {}
		for x in missing:
			temp = []
			lines = np.where(data[x].isnull())[0]
			for l in lines:
				temp.append(ground_data.at[l, x])
			ground_truth[x] = temp
		# get the missing part of the y part(ground truth)
		return data, missing, missing_lines, ground_truth

	def RF(self, data, missing_feature, real_missing_lines, missing_lines, type, num):
		# data of the tuple without missing feature
		# data = pd.get_dummies(data)
		left_features = list(set(data.columns) - set(missing_feature))
		# divide the not missing part into x and y
		training_part = data.drop(missing_lines, axis = 0)
		train_x = training_part[left_features]; train_y = training_part[missing_feature]
		# get the missing part of the x part
		predict_x = data.loc[missing_lines, left_features]
		if type == 'float':
			# model = RandomForestRegressor(oob_score=True, random_state = 1)
			model = RandomForestRegressor(random_state = 1)
			model.fit(train_x, train_y.values.ravel())
			predict_y = model.predict(predict_x)
			# fill the missing parts
			data.loc[missing_lines, missing_feature] = predict_y.round(2)
		else:
			model = RandomForestClassifier(random_state = 1)
			model.fit(train_x, train_y.values.ravel())
			predict_y = model.predict(predict_x)
			data.loc[missing_lines, missing_feature] = predict_y
			if len(predict_y) > 0:
				predict_y = list(num.inverse_transform(predict_y))
		m = dict(zip(missing_lines, predict_y))
		fake = list(set(missing_lines) - set(real_missing_lines))
		fake_prediction = [m[x] for x in fake]
		real_prediction = [m[x] for x in real_missing_lines]
		return data, real_prediction, fake_prediction
		# get the precision, recall and f1-measure of the result

	def Create_Fake_Missing(self, data, missing_count, missing_all):
		xx, yy = np.where(data.notnull())
		length = len(xx) * 0.1
		fake_missing_count = {k:int(length * v /missing_all) for k, v in missing_count.items()}
		cat2num = dict(zip(data.columns, range(len(data.columns))))
		fake_missing = {}
		fake_ground_truth = {}
		for c, n in fake_missing_count.items():
			pos = np.where(yy == cat2num[c])[0]
			x = pos[random.sample([i for i in range(len(pos))], n)]
			fake_ground_truth[c] = data.loc[xx[x], c]
			data.loc[xx[x], c] = None
			fake_missing[c] = x
		return data, fake_missing, fake_ground_truth, fake_missing_count

	def cal_precison(self, ground_truth, prediction, limitation, dt):
			x, y = len(ground_truth), len(prediction)
			if x != y or x == 0:
				print('Length of the two array does not match')
				return -1
			correct = 0
			if dt == 'float':
				for i in range(x):
					if ground_truth[i] - prediction[i] < limitation:
						correct += 1
			else:
				for i in range(x):
					if ground_truth[i] == prediction[i]:
						correct += 1
			return float(correct/x)

	def RF_Missing_Iterative(self, D):
		data, missing, real_missing_lines, real_ground_truth = self.Preprocessing_Iterative(D)
		# ransform the catogory label into numeric label
		num = preprocessing.LabelEncoder()
		real_missing_count = {x:data[x].isnull().value_counts()[1] for x in missing}
		real_missing_count = dict(sorted(real_missing_count.items(), key = lambda x:x[1], reverse = False))
		real_missing_all = sum([v for k, v in real_missing_count.items()])
		# add some missing data artifically
		data, fake_missing, fake_ground_truth, fake_missing_count = self.Create_Fake_Missing(data, real_missing_count, real_missing_all)
		fake_missing_all = sum([v for k, v in fake_missing_count.items()])
		not_missing_lines = {x:np.where(data[x].notnull())[0] for x in missing}
		column_type = {x:data[x].dtype for x in data.columns}
		missing_lines = {x:np.where(data[x].isnull())[0] for x in missing}
		missing_count = {x:data[x].isnull().value_counts()[1] for x in missing}
		missing_count = dict(sorted(missing_count.items(), key = lambda x:x[1], reverse = False))
		missing_all = sum([v for k, v in missing_count.items()])
		num2Cat = {}
		# fill all the missing data with mean
		for x in missing:
			if data[x].dtype == 'float':
				x_mean = data.loc[not_missing_lines[x], x].mean()
				data[x].fillna(x_mean, inplace=True)
			else:
				x_mode = data.loc[not_missing_lines[x], x].mode()
				data[x].fillna(x_mode[0], inplace=True)
		for x in data.columns:
			if data[x].dtype != 'float':
				data[x] = num.fit_transform(data[x])
				# print(num.classes_)
		count = 0
		real_precision = {x:0 for x in missing}
		fake_precision = {x:0 for x in missing}
		limitation = 0.5
		while True:
			for x in missing_count:
				data, real_prediction, fake_prediction = self.RF(data, x, real_missing_lines[x], missing_lines[x], column_type[x], num)
				# data, prediction = XGB(data, x, missing_lines[x], column_type[x], num)
				real_precision[x] = self.cal_precison(real_ground_truth[x], real_prediction, limitation, column_type[x])
				fake_precision[x] = self.cal_precison(list(fake_ground_truth[x]), fake_prediction, limitation, column_type[x])
			real_p = sum([real_precision[x] * real_missing_count[x] for x in missing])/real_missing_all
			fake_p = sum([fake_precision[x] * fake_missing_count[x] for x in missing])/fake_missing_all
			print('missing_for_real:%f' %real_p)
			print('missing_for_fake:%f' %fake_p)
			print('**********')
			count += 1
			if fake_p > 0.9 or count > 10:
				break
		print(real_precision)


if __name__ == '__main__':
	warnings.filterwarnings(action='ignore', category=DeprecationWarning)
	data = pd.read_csv('missing0.3.csv', header = 0, index_col = 0)
	test = RF_Iter_Missing()
	test.RF_Missing_Iterative(data)