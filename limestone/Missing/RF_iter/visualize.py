import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def readData(fileName):
	return pd.read_csv(fileName, header = 0, index_col = 0)

def showMissingNum(data, percent):
	col_miss_count = pd.DataFrame([np.where(data[i].isnull())[0].shape[0]/data.shape[0] for i in data.columns])
	# c = [np.where(data[i].isnull())[0].shape[0]/data.shape[0] for i in data.columns]
	col_miss_count.set_index(data.columns, inplace=True)
	print(col_miss_count.head())
	col_miss_count.plot.bar(colormap = 'Accent')
	plt.title('K = {}'.format(percent))
	plt.ylabel('missing percentage')
	plt.xlabel('columns')
	plt.ylim(0, 1)
	plt.show()

if __name__ == '__main__':
	# folder = 'iris'
	folder = 'shuttle'
	# folder = 'wine'
	percent = 0.5
	F = folder + '/' + str(percent) + '/'
	fname = F + 'missing.csv'
	data = readData(fname)

	showMissingNum(data, percent)