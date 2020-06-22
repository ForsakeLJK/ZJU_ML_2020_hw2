import numpy as np


def logistic(X, y):
	'''
	LR Logistic Regression.

	INPUT:  X: training sample features, P-by-N matrix.
			y: training sample labels, 1-by-N row vector.

	OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
	'''
	P, N = X.shape
	w = np.zeros((P + 1, 1))
	# YOUR CODE HERE
	# begin answer
	learning_rate = 1e-2
	threshold = 1e-8
	max_iters = 1000
	iters = 0
	converted_X = np.vstack((np.ones((1, X.shape[1])), X))

	def getGradient(para):
		'''
		INPUT: para: current parameters, (P+1)-by-1 column vector.

		OUTPUT: gradient: current gradient, (P+1)-by-1 column vector.
		'''
		gradient = np.zeros((P+1, 1))
		# every sample i
		for i in range(N):
			coefficient = - y[0, i] / (1. + np.exp(y[0, i] * np.dot(para.T, converted_X[:, [i]])))
			gradient += coefficient * converted_X[:, [i]]
		return gradient

	for iters in range(max_iters):
		w -= learning_rate * getGradient(w)

		if learning_rate * np.linalg.norm(getGradient(w)) < threshold:
			# print("threshold touched")
			break

	# if(iters == max_iters):
	# 	print("max_iters exceeded")
	# end answer

	return w
