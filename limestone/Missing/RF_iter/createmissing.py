import numpy as np
import pandas as pd
import random
import os
import math
import re

def createMissing(num, data):
	# get the distribution over all the columns of the data
	columns_missing = np.random.rand(data.shape[1])
	columns_missing = columns_missing/sum(columns_missing)
	columns_missing = list(map(lambda x: math.floor(x*num), columns_missing))
	missing_lines = [random.sample(range(0, data.shape[0]), x) for x in columns_missing]
	count = 0
	for column in data.columns:
		data.loc[missing_lines[count], column] = ""
		count += 1
	return data

def createMissing_uniform(num, data):
	columns = data.columns
	col_map = {x:y for x, y in zip(range(len(columns)), columns)}
	missing_count = int(len(columns) * len(data) * num)
	row_idx = list(map(int, np.random.uniform(low = 0, high = len(data), size = missing_count)))
	# row_idx = randint(low = 0, high = len(data), size = missing_count)
	col_idx = list(map(lambda x: col_map[x], list(map(int, np.random.uniform(low = 0, high = len(columns), size = missing_count)))))
	missing_idx = zip(row_idx, col_idx)
	for mi in missing_idx:
		data.at[mi] = None
	return data


def read_from_txt(fname):
	data = np.loadtxt(fname, delimiter = '\s', dtype = 'str')
	temp_data = []
	for x in data:
		temp_data.append(re.split(r'\s+', x))
	return pd.DataFrame(temp_data)

def createFile(fname):
	print('create new file in the directry: {}'.format(fname))
	if not os.path.exists(fname):
		data.to_csv(fname)
		print('create new file done!')
	else:
		print('file have already exists!')
		yes = input('overrider(y/n)?')
		if yes == 'y':
			data.to_csv(fname)
			print('create new file done!')

def createDataSet(data, percent, folder_name):
	# create a new folder structure: new_folder{test.csv, nomissing.csv, missing.csv}
	new_folder_name = folder_name + '/' + str(percent)
	if not os.path.exists(new_folder_name):
		os.makedirs(new_folder_name)

	# create test data
	test_fname = new_folder_name + '/' 'test' + '.csv'
	createFile(test_fname)

	# here we started to generate training data
	data.drop(test_idx, axis = 0, inplace = True)
	data.index =  range(len(data))
	# save the complete data part
	nomissing_fname = new_folder_name + '/' + 'nomissing' + '.csv'
	createFile(nomissing_fname)
	

	feature_num = len(data.columns)
	# create some missing data according to the specified missing rate
	# data = createMissing(int(data.shape[0]*data.shape[1]*percent), data)
	data = createMissing_uniform(percent, data)
	missing_fname = new_folder_name + '/' + 'missing' + '.csv'
	createFile(missing_fname)

if __name__ == '__main__':

	percent = 0.3
	folder_name = 'iris'
	fname = 'iris.csv'
	data = pd.read_csv(folder_name + '/' + fname, header = 0)
	# data = read_from_txt(folder_name + '/' + fname)

	t_size = int(len(data) * 0.25)
	test_idx = random.sample(list(range(len(data))), t_size)
	createDataSet(data, percent, folder_name)
