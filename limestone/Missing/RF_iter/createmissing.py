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

def createMissing_noduplicate(num, data):
	col_len = len(data.columns)
	total_num = len(data) * col_len
	missing_idx_flat = random.sample(range(total_num), int(total_num*num))
	missing_idx = [(int(x/col_len), x%col_len) for x in missing_idx_flat]
	for mi in missing_idx:
		data.iat[mi] = None
	return data

def read_from_txt(fname, missing_token = '', t_pos = -1):
	data = np.loadtxt(fname, delimiter = '\s', dtype = 'str')
	temp_data = []
	if missing_token == '':
		for x in data:
			temp_data.append(re.split(r'\,?\s*', x))
	else:
		for x in data:
			new_tuple = [x if x!=missing_token else None for x in re.split(r'\,?\s*', x)]
			temp_data.append(new_tuple)

	new_data = pd.DataFrame(temp_data)
	if t_pos != -1:
		t_col = new_data.columns[t_pos]
		cols = list(new_data.columns)
		cols.remove(t_col)
		cols.append(t_col)
		new_data = new_data[cols]
	return new_data

def createFile(data, fname):
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

def createDataSet(data, percent, folder_name, test_percent = 0.25):
	# create a new folder structure: new_folder{test.csv, nomissing.csv, missing.csv}
	new_folder_name = folder_name + '/' + str(percent)
	if not os.path.exists(new_folder_name):
		os.makedirs(new_folder_name)

	t_size = int(len(data) * test_percent)

	x = set(np.where(data.isnull())[0])
	complete_idx = list(set(range(len(data))) - x)
	# create test data
	test_idx = random.sample(complete_idx, t_size)
	test_fname = new_folder_name + '/' 'test' + '.csv'
	test_data = data.iloc[test_idx, :]
	createFile(test_data, test_fname)

	# here we started to generate training data
	data.drop(test_idx, axis = 0, inplace = True)
	data.index =  range(len(data))
	# save the complete data part
	nomissing_fname = new_folder_name + '/' + 'nomissing' + '.csv'
	createFile(data, nomissing_fname)
	

	feature_num = len(data.columns)
	# create some missing data according to the specified missing rate
	# data = createMissing(int(data.shape[0]*data.shape[1]*percent), data)
	data = createMissing_noduplicate(percent, data)
	missing_fname = new_folder_name + '/' + 'missing' + '.csv'
	createFile(data, missing_fname)

if __name__ == '__main__':


	percent = 0.5
	folder_name = 'iris'
	fname = 'iris.csv'
	data = pd.read_csv(folder_name + '/' + fname, header = 0)

	# folder_name = 'shuttle'
	# fname = 'shuttle.txt'
	# data = read_from_txt(folder_name + '/' + fname)

	# folder_name = 'yeast'
	# fname = 'yeast.txt'
	# data = read_from_txt(folder_name + '/' + fname)

	# folder_name = 'adult'
	# fname = 'adult.data.txt'
	# missing_token = '?'
	# data = read_from_txt(folder_name + '/' + fname, missing_token)

	# folder_name = 'wine'
	# fname = 'wine.data.txt'
	# data = read_from_txt(folder_name + '/' + fname, 0)

	createDataSet(data, percent, folder_name)
