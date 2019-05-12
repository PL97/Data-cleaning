import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn import preprocessing

def sample(data, size):
    if(size > data.shape[0]):
        print("size greater than the data size!")
        size = data.shape[0]
    idx = random.sample(range(0, data.shape[0]), size)
    return data.iloc[idx, :], idx

def category2num(data):
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
		model = DecisionTreeClassifier(criterion = 'entropy')
		model.fit(feature, label)
		self.model.append(model)
		
	def intergrate(self, feature, label):
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
		result = []
		for m in self.model:
			result.append(m.predict(feature))
		result = np.array(result).transpose()
		return result
	
	def bagging(self, feature):
		new_feature = category2num(feature)[0]
		models_result = self.predict(new_feature)
		result = [pd.Series.mode(x)[0] for x in models_result]
		return result

		
	def stacking(self, feature):
		new_feature = category2num(feature)[0]
		result = self.predict(new_feature)
		new_feature = category2num(pd.DataFrame(result))[0]
		result = self.sec_model.predict(new_feature)
		return result