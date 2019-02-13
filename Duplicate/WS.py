import numpy as np

def Smith_Waterman(str1, str2, s_score, m_score):
	len1, len2 = len(str1), len(str2)
	matrix = np.zeros([len1 + 1, len2 + 1])
	for i in range(len1):
		matrix[i, 0] = 0
	for i in range(len2):
		matrix[0, i] = 0
	Space = 0
	for i in range(1, len1 + 1):
		for j in range(1, len2 + 1):
			Mkj = matrix[i-1, j] - Space
			Mik = matrix[i, j-1] - Space
			Mij = matrix[i-1, j-1] + 1 if str1[i-1].lower() == str2[j-1].lower() else matrix[i-1, j-1] -1
			matrix[i, j] = max(Mij, Mkj, Mik, 0)
	match_str1, match_str2, match_rate = Trace_back(str1, str2, matrix, Space)
	# print(match_str1)
	# print(match_str2)
	# print(match_rate)
	return match_str1, match_str2, match_rate

def Trace_back(str1, str2, M, Space):
	#find max
	x, y = np.where(M == np.max(M))
	x, y = x[0], y[0]
	padding = len(str1) + len(str2) - x - y
	# print(M)
	# print(x, y)
	match_str1, match_str2 = '', ''
	match_count = 0
	score = 0
	count = 0
	while M[x, y] != 0:
		# print(x, y)
		count += 1
		if M[x - 1, y] - Space == M[x, y]:
			x = x -1
			match_str1, match_str2 = str1[x] + match_str1, '_' + match_str2
			score += 0.5
		elif M[x, y - 1] - Space == M[x, y]:
			y = y - 1
			match_str1, match_str2 = '_' + match_str1, str2[y] + match_str2
			score += 0.5
		else:
			x, y = x-1, y-1
			match_str1, match_str2 = str1[x] + match_str1, str2[y] + match_str2
			match_count += 1
			score += 1
		# match_rate = match_count/min(len(str1), len(str2))
	return match_str1, match_str2, score/(len(str1) + len(str2))

if __name__ == '__main__':
	str1 = 'siwtaf'
	str2 = 'Smith_W'
	print(Smith_Waterman(str1, str2, 0.5, 1))
