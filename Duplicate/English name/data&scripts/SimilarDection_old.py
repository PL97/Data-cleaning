import pandas as pd
import numpy as np
import re
from itertools import combinations

class SimilarDection(object):

	def __init__(self, miss, remiss, miss_head, not_miss, not_miss_head):
		self.miss = miss
		self.remiss = remiss
		self.miss_head = miss_head
		self.not_miss = not_miss
		self.not_miss_head = not_miss_head

	def Smith_Waterman(self, str1, str2):
		len1, len2 = len(str1), len(str2)
		matrix = np.zeros([len1 + 1, len2 + 1])
		Space, mismatch = 0.5, 3
		idx1 = [i.start()+1 for i in re.finditer(' ', str1)]
		idx2 = [i.start()+1 for i in re.finditer(' ', str2)]
		idx1.append(0)
		idx2.append(0)
		s = np.ones([len1, len2])
		for i in idx1:
			for j in idx2:
				s[i, j] = 3
		# define reward matrix
		for i in range(1, len1 + 1):
			for j in range(1, len2 + 1):
				Mkj = max([matrix[i-k, j] - Space * k for k in range(1, max(i, 2))])
				Mik = max([matrix[i, j-k] - Space * k for k in range(1, max(2, j))])
				Mij = matrix[i-1, j-1] + s[i-1, j-1] if str1[i-1].lower() == str2[j-1].lower() else matrix[i-1, j-1] - s[i-1, j-1]
				matrix[i, j] = max(Mij, Mkj, Mik, 0)
		print(matrix)
		match_str1, match_str2, match_rate = self.Trace_back(str1, str2, matrix, Space, idx1, idx2, s)
		print(matrix)
		return match_str1, match_str2, match_rate

	def Trace_back(self, str1, str2, M, Space, idx1, idx2, s):
		#find max
		x, y = np.where(M == np.max(M))
		x, y = x[0], y[0]
		match_str1, match_str2 = '', ''
		match_count = 0
		score, count = 0, 0
	    # find all the character at the begining of the string
		idx3 = [i.start()+1 for i in re.finditer('.', str1)]
		idx4 = [i.start()+1 for i in re.finditer('.', str2)]
	    # punish all the character did not match
		punish = min(len(str1)-x, len(str2) - y)
		flag = 0
		while M[x, y] != 0:
		# while x >= 1 and y >= 1:
			count += 1
			print(x, y)
			print(s[x-1, y-1])
			# print(flag)
			if M[x-1, y-1] + s[x-1, y-1] == M[x, y]:
				print('1')
				x, y = x-1, y-1
				match_str1, match_str2 = str1[x] + match_str1, str2[y] + match_str2
				match_count += 1
				if x in idx1 and y in idx2:
					score += self.not_miss_head
				else:
					score += self.not_miss
				flag = 0
			elif M[x-1, y-1] - s[x-1, y-1] == M[x, y]:
				print('2')
				x, y = x -1, y - 1
				match_str1, match_str2 = '/' + match_str1, '/' + match_str2
				if x in idx1 and y in idx2:
					score -= self.miss_head
				else:
					score -= self.remiss*flag if flag!=0 else self.miss
				flag = True
			elif M[x - 1, y] - Space == M[x, y]:
				print('3')
				x = x -1
				match_str1, match_str2 = str1[x] + match_str1, '_' + match_str2
				# if y in idx4:
				# 	continue
				if x in idx1 and y in idx2:
					score -= self.miss_head
				else:
					score -= self.remiss*flag if flag!=0 else self.miss
				flag += 1
			else:
				print('4')
				y = y - 1
				match_str1, match_str2 = '_' + match_str1, str2[y] + match_str2
				# if x in idx3:
				# 	continue
				if x in idx1 and y in idx2:
					score -= self.miss_head
				else:
					score -= self.remiss*flag if flag!=0 else self.miss
				flag += 1
			# match_rate = match_count/min(len(str1), len(str2))
			print(score)
			# print(match_str1, match_str2)
		return match_str1, match_str2, (2*score-punish)/(len(str1) + len(str2))


def read_file(fname):
	return pd.read_csv(fname, index_col = 0)


if __name__ == '__main__':
	# fname = 'raw_data_location.csv'
	# # fname = 'full_name.csv'
	# data = read_file(fname)
	# score1 = []
	# test = SimilarDection(0.5, 1, 2, 1, 2)
	# for row in data.itertuples(index=False):
	# 	score1.append(test.Smith_Waterman(row[0], row[1])[2])
	# 	# print(Smith_Waterman(str1, str2)
	# print(score1)

	str1 = 'Hgn. M'
	str2 = 'Hxn M'
	print(str1)
	print(str2)
	# miss, remiss, miss_head, not_miss, not_miss_head
	test = SimilarDection(0.5, 1, 2, 1, 2)
	print(test.Smith_Waterman(str1, str2))