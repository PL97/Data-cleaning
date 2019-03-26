import numpy as np
from scipy.misc import derivative


def f(x): 
	theta = np.array([1, 2, 3, 4])
	return np.dot(x, theta)
	# return x ** 2


x = [1, 2, 3, 4]
print(derivative(f, x, dx=1e-6))