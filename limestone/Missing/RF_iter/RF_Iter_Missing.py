import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import warnings
import random
from sklearn.tree import DecisionTreeClassifier
import operator
from sklearn.ensemble import IsolationForest
from BitVector import BitVector
import copy


class RF_Iter_Missing(object):

	def __init__(self, model, precision):
		self.model = model
		self.precision = precision

	def Preprocessing_Iterative(self, data):
		# # prepare for the data
		# data = pd.read_csv('missing0.3.csv', header = 0, index_col = 0)
		missing = [x for x in data.columns if data[x].isnull().any()]
		# the index of tuple with some missing features
		missing_lines = {x:np.where(data[x].isnull())[0] for x in missing}
		missing_count = {x:data[x].isnull().value_counts()[1] for x in missing}
		missing_count = dict(sorted(missing_count.items(), key = lambda x:x[1], reverse = False))
		return data, missing, missing_lines, missing_count

	def RF(self, data, missing_feature, missing_lines, type):
		""" Ramdom Forest used in the missing attribute filling, which is uesd in the 
		iterative process

		Parameters
		----------
		data: DataFrame, raw data contains some missing values
		missing_features: String-like, the column(only one column) that contains misssing values
		missing_lines: List or array, all the lines where those missing values located
		type: type of missing attribute

		Returns
		-------
		data: DataFrame, revised data without missing values in the specified feature
		predict_y: array, missing values predicted by the model
		"""
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
			model = RandomForestClassifier(random_state = 2)
			model.fit(train_x, train_y.values.ravel())
			predict_y = model.predict(predict_x)
			data.loc[missing_lines, missing_feature] = predict_y
			if len(predict_y) > 0:
				predict_y = list(predict_y)
		return data, predict_y
		# get the precision, recall and f1-measure of the result

	def RF_Missing_Iterative(self, D, t_D, model, predict_type = 0):
		""" The method used in the a Iterative process for missing and outlier detection
		and fix. 

		First by using Iforest to detect some outlier tuples, and then set those values
		as missing like tuple.

		After that, using average of the golbal data on each attribute to fill the
		corresponding position. If it is category-like attribute then apply it with mode.

		Then applied the Random Forest on the data, and try to fill the missing data with
		the model predict values. Since there are no observing missing values in the data
		(since it has been fill by either mode and mean of the global data).

		Parameters
		----------
		D: DataFrame, data with missing values
		D_t: DataFrame, data without missing values and uesd for cleaned data performance 
		testing

		Returns
		-------
		data: DataFrame, data which contains no missing values and has satified the specified 
		classifying or clustering precision requirement.
		"""
		data, missing, missing_lines, missing_count = self.Preprocessing_Iterative(D)
		# add some missing data artifically
		not_missing_lines = {x:np.where(data[x].notnull())[0] for x in data.columns}

		# get the lines of the complete data
		outlier_features = D.columns[:-1]
		selected_data = data[outlier_features]
		missing_row_idx = set(np.where(selected_data.isnull())[0])
		complete_row_idx = list(set(range(len(data))) - missing_row_idx)

		# train the outlier dection on the complete data
		# print(missing_lines)

		column_type = {x:data[x].dtype for x in data.columns}
		# fill all the missing data with mean
		data, fill_data = self.fill_missing(data, not_missing_lines, predict_type)
		# transform all the category attribute to numeric ones
		data, t_D, num2Cat = self.category2num(data, t_D)

		for k, v in num2Cat.items():
			# fill_data[k]= dict(zip(v.values(),v.keys()))[fill_data[k]]
			fill_data[k] = v.transform([fill_data[k]])[0]

		# get test_data and split it into features and label
		X_test = t_D.drop(t_D.columns[-1], axis = 1)
		y_test = t_D.loc[:, t_D.columns[-1]]

		accuracy = self.evaluate(model, data, X_test, y_test)
		print('origional precision {}'.format(accuracy))

		if predict_type == 1:
			column_type[data.columns[-1]] = 'int'

		# train for the outlier dection model
		# ilf, missing_lines = self.training_oulier_localdata(data, outlier_features, complete_row_idx, missing_lines)
		# ilf = self.training_oulier_testdata(t_D, outlier_features)
		ilf, missing_lines = self.training_oulier_alldata(data, t_D, outlier_features, complete_row_idx, missing_lines)
		p = [0] * 3
		count = 0
		best_fit = []
		best_precision = 0
		while True:
			pre_predict = []
			for x in missing_count:
				pre_predict.extend(list(data.loc[missing_lines[x], x]))
			post_predict = []
			for x in missing_count:
				data, prediction = self.RF(data, x, missing_lines[x], column_type[x])
				post_predict.extend(prediction)
			count += 1
			if count % 3 == 0:
				data = self.outlier_fix(data, ilf, outlier_features, missing_lines, fill_data)
			accuracy = self.evaluate(model, data, X_test, y_test)
			# save the best fit result
			if accuracy > best_precision:
				print('update')
				best_fit = post_predict
				best_precision = accuracy
			p[(count-1)%3] = accuracy
			print('Iteration {}, precision is {}'.format(count, accuracy))
			# stop = operator.eq(post_predict, pre_predict) # too strict
			loss = sum(abs(np.array(post_predict) - np.array(pre_predict)))/len(pre_predict) if len(pre_predict) != 0 else 10
			print('current matrix change is {}'.format(loss))
			# loop stop condition
			if count > 30 or accuracy > self.precision or loss < 0.00001:
				break
			if p[0] == p[1] and p[1] == p[2]:
				break
		# recover the data with the best fit
		temp_count = 0
		for x in missing_count:
			data.loc[missing_lines[x], x] = best_fit[temp_count:temp_count+len(missing_lines[x])]
			temp_count += len(missing_lines[x])
		# recover all numeric attributes to category attributes
		for k, v in num2Cat.items():
			data[k] = v.inverse_transform(data[k])
		return data

	def evaluate(self, model, data, X_test, y_test):
		""" evaluate the performance of the cleaned data on the test data set with the
		user specifying the model(machine learning model)

		Parameters
		----------
		model: machine learning model, which can fit and predict
		data: DataFrame, cleaned data
		X_test: DataFrame, test data's feature
		y_test: DataFrame, test data's label

		Returns
		-------
		accuracy: float, predict accuracy
		"""
		X_train = data.drop(data.columns[-1], axis = 1)
		y_train = data.loc[:, data.columns[-1]]
		model.fit(X_train, y_train)
		y_predict = model.predict(X_test)
		accuracy = len(np.where(y_test == y_predict)[0])/len(y_test)
		return accuracy

	def fill_missing(self, data, not_missing_lines, predict_type):
		""" fill all the missing data with mean or mode data, the global data are
		collected from the tuple-complete ones

		Parameters
		----------
		data: DataFrame, data that contains some missing position
		not_missing_lines: dict, key is the name of each columns, values are the 
		lines that are not missing

		Returns
		-------
		data: DataFrame, data that has been cleaned without missing items
		"""
		fill_data = {}
		for x in data.columns:
			if data[x].dtype == 'float':
				if x == data.columns[-1] and predict_type == 1:
					fill_data[x] = data.loc[not_missing_lines[x], x].mode()[0]
				else:
					fill_data[x] = data.loc[not_missing_lines[x], x].mean()
			else:
				fill_data[x] = data.loc[not_missing_lines[x], x].mode()[0]
			data[x].fillna(fill_data[x], inplace=True)
		return data, fill_data

	def outlier_fix(self, data, ilf, outlier_features, missing_lines, fill_data):
		""" detect the outlier data and fills it with either mode and mean of the global data
		Here we only focus on those missing position, other data are not put into consideration

		Parameters
		----------
		data: DataFrame, data which waited to be detected
		ilf: outlier dection model
		outlier_features: list, features that used for outlier dection

		Returns
		-------
		data: DataFrame, data that has repaced the outlier data with either mode(categorical)
		or mean(numeric)
		"""
		pred = ilf.predict(data[outlier_features])
		outlier_idx = np.where(pred == -1)[0]
		bitarray = BitVector(size = len(data))
		count = 0
		for k, v in missing_lines.items():
			bitarray.reset(val = 0)
			for i in v:
				# then the i is too large it will converet to float
				bitarray[int(i)] = 1
			for j in outlier_idx:
				if bitarray[j] == 1:
					data.at[j, k] = fill_data[k]
					count += 1
		print('revised {} items'.format(count))
		return data

	def tuning(self):
		pass

	def training_oulier_localdata(self, data, outlier_features, complete_row_idx, missing_lines):
		complete_data = data[outlier_features].iloc[complete_row_idx, :]
		ilf = IsolationForest(n_estimators=min(100, len(complete_data)), n_jobs=-1, verbose=2)
		ilf.fit(complete_data)
		pred = ilf.predict(complete_data)
		outlier_idx = [complete_row_idx[i] for i in np.where(pred == -1)[0]]
		# put these outlier data into the missing list
		for k, v in missing_lines.items():
			missing_lines[k] = np.append(v, outlier_idx)
		return ilf, missing_lines

	def training_oulier_testdata(self, data, outlier_features):
		ilf = IsolationForest(n_estimators=min(100, len(data)), n_jobs=-1, verbose=2)
		ilf.fit(data[outlier_features])
		return ilf

	def training_oulier_alldata(self, data, t_data, outlier_features, complete_row_idx, missing_lines):
		complete_data = data[outlier_features].iloc[complete_row_idx, :]
		test_data = t_data[outlier_features]
		all_data = pd.concat([complete_data, test_data], axis = 0, ignore_index=True)
		ilf = IsolationForest(n_estimators=min(100, len(all_data)), n_jobs=-1, verbose=2)
		ilf.fit(all_data)
		if len(complete_data) > 0:
			pred = ilf.predict(complete_data)
			outlier_idx = [complete_row_idx[i] for i in np.where(pred == -1)[0]]
			# put these outlier data into the missing list
			for k, v in missing_lines.items():
				missing_lines[k] = np.append(v, outlier_idx)
		return ilf, missing_lines

	def category2num(self, data, t_D):
		num2Cat = {}
		for x in data.columns:
			if data[x].dtype != 'float' and data[x].dtype != 'int64':
				num = preprocessing.LabelEncoder()
				temp = pd.concat([data[x], t_D[x]], axis = 0)
				num.fit(temp.astype(str))
				data[x] = num.transform(data[x].astype(str))
				t_D[x] = num.transform(t_D[x].astype(str))
				num2Cat[x] = num
				# data[x] = num.fit_transform(data[x])
				# num2Cat[x] = dict(enumerate(num.classes_))
		return data, t_D, num2Cat

def complete_data_evaluate(test, nomiss_data, test_data):
	c_test_data = copy.copy(test_data)
	# c_data = copy.copy(data)
	test.category2num(nomiss_data, c_test_data)
	# test.category2num(c_data, c_test_data)
	X_train = c_test_data.drop(data.columns[-1], axis = 1)
	y_train = c_test_data.loc[:, data.columns[-1]]
	p = test.evaluate(model, nomiss_data, X_train, y_train)
	return p

if __name__ == '__main__':
	percent = 0.5
	# name = 'wine'
	name = 'iris'
	# name = 'shuttle'
	# name = 'yeast'
	# name = 'adult'
	folder = name + '/' + str(percent)


	missing_file = 'missing' + '.csv'
	test_file = 'test' + '.csv'
	nomissing_file = 'nomissing' + '.csv'

	warnings.filterwarnings(action='ignore', category=DeprecationWarning)
	data = pd.read_csv(folder + '/' + missing_file, header = 0, index_col = 0)
	test_data = pd.read_csv(folder + '/' + test_file, header = 0, index_col = 0)
	nomiss_data = pd.read_csv(folder + '/' + nomissing_file, header = 0, index_col = 0)
	model = DecisionTreeClassifier(criterion = 'entropy')
	# model = RandomForestRegressor(random_state = 1)

	test = RF_Iter_Missing(model, 0.99)
	# complete_data_evaluate(test_data)
	p = complete_data_evaluate(test, nomiss_data, test_data)
	print('no missing precision is {}'.format(p))
	new_data = test.RF_Missing_Iterative(data, test_data, model)
	# new_data = test.RF_Missing_Iterative(data, test_data, model, 1)

	recover_file = 'recover_data.csv'
	new_data.to_csv(folder + '/' + recover_file)


