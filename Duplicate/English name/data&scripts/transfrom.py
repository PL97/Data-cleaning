import pandas as pd
import numpy as np
import re

def read_file(fname):
	data = np.loadtxt(fname, dtype = 'str', delimiter='\n')
	return data

# format xxx chinese xxx
# extract xxx xxx
def process1(data):
	result = []
	for d in data:
		# l = re.split(r'[a-z|A-Z]+', d)
		pattern = re.compile(r'([a-zA-z]+(\s[a-zA-z]+)*)')   # 查找数字
		temp = np.array(pattern.findall(d))
		result.append(temp[:, 0])
	result = pd.DataFrame(result)
	return result

fname = 'raw_data_location.txt'
data = read_file(fname)
new_data = process1(data)
new_data.to_csv(fname.split('.')[0] + '.csv')

# test = ' West Virginia 缩写：WV '
# p = re.findall(r'([a-zA-z]+(\s[a-zA-z]+)*)', test)

# # t = 'eababcc ec'
# # p = re.findall(r'(e(ab)*c)', t)
# p = np.array(p)
# print(p[:, 0])