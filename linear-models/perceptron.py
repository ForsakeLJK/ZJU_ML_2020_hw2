import numpy as np

def perceptron(X, y):
	'''
	PERCEPTRON Perceptron Learning Algorithm.

		INPUT:  X: training sample features, P-by-N matrix.
				y: training sample labels, 1-by-N row vector.

		OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
				iter: number of iterations

	'''
	P, N = X.shape
	w = np.zeros((P + 1, 1))
	iters = 0
	findSolutionVector = False
	# convert X, (P+1)-by-N matrix
	converted_X = np.vstack((np.ones((1, X.shape[1])), X))
	max_iters = 6000
	# YOUR CODE HERE
	# begin answer
	while not findSolutionVector:
		errorOccur = False

		for i in range(N):
			# fetch a sample 
			sample = converted_X[:, [i]]
			# print("sample:{}".format(sample))
			# predict this sample 
			y_predict = np.sign(np.dot(w.T, sample))
			# predict error, then update w
			if(y_predict[0, 0] != y[0, i]):
				w += sample * y[0, i]
				errorOccur = True
			# increment iters
			iters += 1
		


		# all predictions are correct, break
		if not errorOccur:
			findSolutionVector = True
		else:
			pass
			# print("y_predict: {}, y: {}, w: {}, sample: {}".format(y_predict[0, 0], y[0, i], w, sample))

		# too many iterations, break
		if iters >= max_iters:
			# print("max_iters exceeded")
			break
	# end answer

	return w, iters
