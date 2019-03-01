import pandas as pd
import numpy as np
import re
from itertools import combinations

class SimilarDection(object):

    def __init__(self, head, ordinary_match, space, abbr):
        self.head = head
        self.space = space
        self.ordinary_match = ordinary_match
        self.abbr = abbr


    def Smith_Waterman(self, str1, str2):
        if len(str1) > len(str2):
            str1, str2 = str2, str1
        len1, len2 = len(str1), len(str2)
        matrix = np.zeros([len1 + 1, len2 + 1])
        # find all the character at the begining of the string
        idx1 = [i.start()+1 for i in re.finditer(' ', str1)]
        idx2 = [i.start()+1 for i in re.finditer(' ', str2)]
        idx1.append(0)
        idx2.append(0)
        # find all .s
        idx3 = [i.start() for i in re.finditer('\.', str1)]
        idx4 = [i.start() for i in re.finditer('\.', str2)]
        s = np.ones([len1, len2])
        s[idx3, :] = self.abbr
        s[:, idx4] = self.abbr
        for i in idx1:
            for j in idx2:
                s[i, j] = self.head
        # define reward matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                Mkj = max([matrix[i-k, j] - self.space * k for k in range(1, max(i, 2))])
                Mik = max([matrix[i, j-k] - self.space * k for k in range(1, max(2, j))])
                Mij = matrix[i-1, j-1] + s[i-1, j-1] if str1[i-1].lower() == str2[j-1].lower() else matrix[i-1, j-1] - s[i-1, j-1]
                matrix[i, j] = max(Mij, Mkj, Mik, 0)
                matrix[i, j] = round(matrix[i, j], 2)
        match_str1, match_str2, match_rate = self.Trace_back(str1, str2, matrix, s, idx3, idx4)
        return match_str1, match_str2, match_rate

    def Trace_back(self, str1, str2, M, s, index3, index4):
        #find max
        x, y = np.where(M == np.max(M))
        x, y = x[0], y[0]
        match_str1, match_str2 = '', ''
        match_count = 0
        score, count = 0, 0
        # punish all the character did not match
        punish = min(len(str1)-x, len(str2) - y)
        flag = 0
        while x >= 1 and y >= 1:
            count += 1
            if M[x-1, y-1] + s[x-1, y-1] == M[x, y] and str1[x-1] == str2[y-1]:
                x, y = x-1, y-1
                match_str1, match_str2 = str1[x] + match_str1, str2[y] + match_str2
                match_count += 1
                score += s[x, y]
                flag = 0
            elif round(M[x, y - 1] - self.space, 2) == M[x, y]:
                y = y - 1
                match_str1, match_str2 = '_' + match_str1, str2[y] + match_str2
                score -= s[x-1, y] if flag == 0 else s[x-1, y] * (flag+1)
                flag += 1
            elif round(M[x - 1, y] - self.space, 2) == M[x, y]:
                x = x -1
                match_str1, match_str2 = str1[x] + match_str1, '_' + match_str2
                score -= s[x, y-1] if flag == 0 else s[x, y-1] * (flag+1)
                flag += 1
            else:
                x, y = x -1, y - 1
                match_str1, match_str2 = '/' + match_str1, '/' + match_str2
                score -= s[x, y] if flag == 0 else s[x, y] * (flag+1)
                flag += 1
        return match_str1, match_str2, (2*score-punish)/(len(str1) + len(str2))

def read_file(fname):
    return pd.read_csv(fname, index_col = 0)


if __name__ == '__main__':
    str1 = 'A. S'
    str2 = 'Arsdf S'
    print(str1)
    print(str2)
    # head, ordinary_match, space, abbr
    test = SimilarDection(2, 1, 0.5, 0.1)
    print(test.Smith_Waterman(str1, str2))