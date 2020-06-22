import numpy as np

def ridge(X, y, lmbda):
	'''
	RIDGE Ridge Regression.

		INPUT:  X: training sample features, P-by-N matrix.
				y: training sample labels, 1-by-N row vector.
				lmbda: regularization parameter.

		OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

	NOTE: You can use pinv() if the matrix is singular.
	'''
	P, N = X.shape
	w = np.zeros((P + 1, 1))
	# YOUR CODE HERE
	# begin answer
	converted_X = np.vstack((np.ones((1, X.shape[1])), X))

	# (P+1)-by-(P+1) square
	tmp2 = np.matmul(converted_X, converted_X.T)

	# (P+1) identity
	id_mat = np.identity(tmp2.shape[0])

	# (P+1)-by-N matrix
	tmp1 = np.matmul(np.linalg.pinv(tmp2 +
                                lmbda * id_mat), converted_X)

	w = np.matmul(tmp1, y.T)
	# end answer
	return w
