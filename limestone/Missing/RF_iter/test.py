from RF_Iter_Missing import *
from DTClassifier import *
import numpy as np
import pandas as pd
import warnings
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import copy


if __name__ == '__main__':
	percent = 0.1
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
	p = complete_data_evaluate(model, test, nomiss_data, test_data)

	test_times = 3
	p_all = {}; p_fill = {}; p_clean = {}
	for i in range(test_times):
		print('no missing precision is {}'.format(p))
		temp_data = copy.deepcopy(data)
		temp_test_data = pd.read_csv(folder + '/' + test_file, header = 0, index_col = 0)
		new_data, p_all[i], p_fill[i]  = test.RF_Missing_Iterative(temp_data, temp_test_data, model, 3, max_iter = 10, max_diff = 0.001, predict_type = 1)
		# new_data, p_all[i], p_fill[i]  = test.RF_Missing_Iterative(temp_data, temp_test_data, model, 4, 10, 1)
		p_clean[i] = TranATest(new_data, temp_test_data)


	colors = ['cyan', 'coral', 'orange', 'grey']
	shapes = ['v', 'o', 'x']
	plt.figure()
	plt.title('K = {}'.format(percent))
	plt.xlabel('iteration')
	plt.ylabel('precision')
	for i in range(test_times):
		plt.plot(p_all[i], c = colors[i])
		plt.scatter(len(p_all[i]), p_clean[i], marker = shapes[i], c = 'green')
		plt.scatter(0, p_fill[i], marker = shapes[i], c = 'red')
	plt.show()
