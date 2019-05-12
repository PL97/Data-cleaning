import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn import preprocessing

def sample(data, size):
	"""
	sampling from the input data wich specified size

	Parameters:
	-----------
	data: DataFrame, data waited for sampling
	size: int, sample size

	Returns:
	--------
	data: DataFrame, sampled data
	idx: list, the idx of the sampled data
	"""

	if(size > data.shape[0]):
		print("size greater than the data size!")
		size = data.shape[0]
	idx = random.sample(range(0, data.shape[0]), size)
	return data.iloc[idx, :], idx

def category2num(data):
	"""
	transform the input data with all numerical type columns

	Parameters:
	-----------
	data: DataFrame, the input data

	Returns:
	--------
	data: DataFrame, data after transformation
	num2Cat: dict, key:columns' name, value: labelEncoder, the encoder
	which can transfrom the data from categorical one to numerical one
	"""

	num2Cat = {}
	for x in data.columns:
		if data[x].dtype != 'float' and data[x].dtype != 'int64':
			num = preprocessing.LabelEncoder()
			num.fit(data[x].astype(str))
			data[x] = num.transform(data[x].astype(str))
			num2Cat[x] = num
	return data, num2Cat

# simple stacking
class ensembleLearning:
	def __init__(self):
		self.model = []
		pass
	
	def trainForSub(self, feature, label):
		""" train for the sub classifier, each time we call this function we 
		can get a sub-classifier

		Parameters:
		-----------
		feature: DataFrame, the data that only contains features
		label: Series, the label

		Returns:
		--------
		NULL

		"""
		model = DecisionTreeClassifier(criterion = 'entropy')
		model.fit(feature, label)
		self.model.append(model)
		
	def intergrate(self, feature, label):
		""" used in the stacking method. We apply all the sub-classifier together and 
		use the output of the all the classifier as features and the real label as 
		output. Then we get a higher level classifier

		Parameters:
		-----------
		feature: DataFrame, the data that only contains features
		label: Series, the label

		Returns:
		--------
		NULL
		"""
		result = []
		for m in self.model:
			result.append(m.predict(feature))
		result = np.array(result).transpose()

		model = DecisionTreeClassifier(criterion = 'entropy')
		new_feature = category2num(pd.DataFrame(result))[0]
		new_label = np.array(label)
		model.fit(new_feature, new_label)
		self.sec_model = model
	
	def predict(self, feature):
		""" gather all the result from all the sub-classifier and return the intergrated result

		Parameters:
		-----------
		feature: DataFrame, features used for prediction

		Returns:
		result: array, the intergrated result
		"""

		result = []
		for m in self.model:
			result.append(m.predict(feature))
		result = np.array(result).transpose()
		return result
	
	def bagging(self, feature):
		""" ensemble method——bagging

		Parameters:
		-----------
		feature: DataFrame, features used for prediction

		Returns:
		result: list, the result predicted by the bagging model(vote)
		"""
		new_feature = category2num(feature)[0]
		models_result = self.predict(new_feature)
		result = [pd.Series.mode(x)[0] for x in models_result]
		return result

		
	def stacking(self, feature):
		""" ensemble method——stacking

		Parameters:
		-----------
		feature: DataFrame, feature used for prediction

		Returns:
		--------
		result: list, the result predicted by the stacking model(to be more specific
		it is produced by the high level classifier)
		"""
		new_feature = category2num(feature)[0]
		result = self.predict(new_feature)
		new_feature = category2num(pd.DataFrame(result))[0]
		result = self.sec_model.predict(new_feature)
		return result