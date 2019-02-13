import numpy as np
import pandas as pd
import copy

def read_file(fname):
	return pd.read_csv(fname, index_col = 0)

fname = 'raw_data_location.csv'
data = read_file(fname)

new_string = [''] * len(data)
count = 0
for row in data.itertuples(index = False):
	string = row[1]
	if string[0] == 'A':
		new_string[count] = 'B' + string[1:]
	else:
		new_string[count] = 'A' + string[1:]
	count += 1

full_name = list(data.iloc[:, 0])

# new_data = pd.DataFrame(np.vstack((full_name, new_string)).transpose())
new_data = pd.DataFrame(new_string)

merged_data = pd.concat([data, new_data],axis=1,ignore_index=True)

merged_data.to_csv('similar.csv')