import numpy as np
from BitVector import BitVector

class BloomFilter_M(object):
	"""docstring for BloomFilter_M"""
	''' 
	m is the length of the bitvector
	n is the number of the key
	k is the number of the hash function
	p is positive-false rate
	'''
	def __init__(self, n, p):
		self.m = int(-(n*np.log(p))/(np.log(2)**2))
		self.bitarray = BitVector(size = self.m)
		self.markarry = BitVector(size = self.m)
		self.n = n
		self.k = int(-np.log2(p))
		self.p = p

	def generate_seeds(self, num):
		x = 131
		count = 0
		while count < num:
			yield x
			x = x*10+3 if count%2 == 0 else x*10+1
			count += 1

	# hash function
	def BKDRHash(self, string, seed):
		# seed = 131
		my_hash = 0
		for ch in string:
			my_hash = my_hash * seed + ord(ch)
		return my_hash % self.m

	def SetArray(self, L):
		for l in L:
			seed = self.generate_seeds(self.k)
			for s in seed:
				temp_index = self.BKDRHash(str(l), s)
				if self.bitarray[temp_index] == 1:
					self.markarry[temp_index] = 1
				else:
					self.bitarray[temp_index] = 1

	def findDuplicate(self, L):
		self.SetArray(L)
		duplicate = {}
		indexs = [-1]*self.k
		for l in L:
			seed = self.generate_seeds(self.k)
			n = 0
			for s in seed:
				flag = True
				temp_index = self.BKDRHash(str(l), s)
				indexs[n] = temp_index
				n += 1
				if self.markarry[temp_index] == 0:
					flag = False
					break
			if flag:
				duplicate[l] = indexs
		duplicate_keys = {x:[] for x in duplicate.keys()}
		for i in range(len(L)):
			if L[i] in duplicate.keys():
				duplicate_keys[L[i]].append(i)
		return duplicate_keys

	def find_duplicate_by_column(self, data, columns):
		if set(data.columns) >= set(columns):
			temp_data = [[str(y) for y in list(x)] for x in data.loc[:, columns].itertuples()]
			new_data = ['-'.join(x[1:]) for x in temp_data]
			result = self.findDuplicate(new_data)
			duplicate_pairs = list(result.values())
			idx = [j for i in duplicate_pairs for j in i[1:]]
			new_data = data.drop(idx, axis = 0)
			print(new_data.shape)
		else:
			print('input columns are not contained in the data')
			duplicate_pairs = None
		return new_data

if __name__ == '__main__':
	test = BloomFilter_M(10, 0.0001)
	L = ['Le', 'pang', 'Le', 'test', 'tet', 'text', 'tett', 'test', 'png', 'text', 'png']
	print(test.findDuplicate(L))