from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from sklearn import preprocessing

ilf = IsolationForest(n_estimators=100, n_jobs=-1, verbose=2)

data = pd.read_csv('iris.csv', header = 0, index_col = 0)
features = data.columns

num = preprocessing.LabelEncoder()
for x in data.columns:
	if data[x].dtype != 'float':
		data[x] = num.fit_transform(data[x])

ilf.fit(data)
pred = ilf.predict(data)

print(pred)