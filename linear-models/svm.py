import numpy as np
from scipy.optimize import minimize

def svm(X, y):
	'''
	SVM Support vector machine.

	INPUT:  X: training sample features, P-by-N matrix.
			y: training sample labels, 1-by-N row vector.

	OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
			num: number of support vectors

	'''
	P, N = X.shape
	w = np.zeros((P + 1, 1))
	num = 0
	converted_X = np.vstack((np.ones((1, X.shape[1])), X))

	# YOUR CODE HERE
	# Please implement SVM with scipy.optimize. You should be able to implement
	# it within 20 lines of code. The optimization should converge wtih any method
	# that support constrain.
	# begin answer
	def loss_func(param):
		norm_w = np.linalg.norm(param)
		# maximum margin classifier
		return 0.5 * norm_w * norm_w

	def constraint(param):
		tmp = np.ones((1, N))
		# constraint:   y_i * (w.T * x) >= 1
		# convert to matrix notation
		return (y * np.matmul(param.T, converted_X) - tmp).reshape(N)
	
	res = minimize(loss_func, w, constraints=({'type': 'ineq', 'fun': constraint}))
	w = res.x
	
	distanceToMin = 1e-5

	# every point
	for i in range(N):
		if np.abs(y[0, i] * np.dot(w.T, converted_X[:, [i]])) - 1 <= distanceToMin:
			num += 1
	# end answer
	return w, num

