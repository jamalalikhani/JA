# ===============================================================
'''
These Machine Learning Algorithms are implemented from scratch by
Jamal Alikhani (March 2017)
for learning purposes
'''
# ===============================================================

import numpy as np
import random
import copy

# ===============================================================
# Naive Bayes (Multinomial):
# ===============================================================

class NaiveBayes():
	def __init__(self):
		pass

	def get_classes(self, lables):
		n = len(lables)
		self.classes = np.unique(lables)

	def fit(self, features, lables):
		self.get_classes(lables)

		N = features.shape
		self.mu = {}
		self.std = {}	
		self.prior = {}
		for c in self.classes:
			self.prior[c] = len(lables[lables==c])/len(lables)
			
			# Baussian distr over each feature
			mean_c = np.mean(features[lables==c],axis=0)
			std_c = np.std(features[lables==c],axis=0)
			self.mu[c] = mean_c
			self.std[c] = std_c
	
	def predict(self, features_test):
		
		n_cls = len(self.classes)
		n = len(features_test)
		pred = np.zeros(n)
		P_x={}
		P_x_post = {}
		Posterior = {}
		for i in range(n):

			denominator = 0
			for c in self.classes:
				P_x[c] = np.exp(-0.5*(features_test[i]-self.mu[c])**2\
					/self.std[c]**2)/(self.std[c])
				P_x_post[c] = P_x[c].prod() * self.prior[c]
				denominator += P_x_post[c]

			for c in self.classes:
				Posterior[c] = P_x_post[c] / denominator
				if Posterior[c] >= 1/n_cls:
					pred[i] = c
					break
		return pred

	def score(self, feature_test, lables_test):
		y_pred = self.predict(feature_test)
		correct = np.equal(y_pred,lables_test)
		accuracy = np.sum(correct)/len(y_pred)
		
		return accuracy

# ===============================================================
# KMeans (multinomial): 
# ===============================================================

class KMeans():
	def __init__(self, n_clusters, max_iter=1000):
		self.n_clusters = n_clusters
		self.max_iter = max_iter

	def fit(self, x):
		# k_mean clustring over train set
		self.mu_clstr = []
		
		E = np.zeros(self.n_clusters)
		n = len(x)
		# initialization
		clstr_lable = np.array(np.zeros(n), dtype='int32')
		a = x[0]
		for c in range(self.n_clusters):
			self.mu_clstr.append(x[c])
		self.mu_clstr = np.array(self.mu_clstr)
		
		e = 1
		m = 0
		while e > 0 or m>self.max_iter:
			clstr_lable = np.array(np.zeros(n), dtype='int32')		

			for i in range(n):
				dist = np.zeros(self.n_clusters)
				for c in range(self.n_clusters):
					dist[c] = np.linalg.norm(x[i]-self.mu_clstr[c])
				clstr_lable[i] = np.argmin(dist)

			for c in range(self.n_clusters):
				a = np.mean(x[clstr_lable==c], axis=0)			
				E[c] = np.linalg.norm(self.mu_clstr[c] - a)
				
				self.mu_clstr[c] = a 

			e = np.max(E)
			m += 1

	def predict(self, x_test):
		n = len(x_test)
		clstr_lable = np.array(np.zeros(n), dtype='int32')
	
		for i in range(n):
			dist = np.zeros(self.n_clusters)
			for c in range(self.n_clusters):
				dist[c] = np.linalg.norm(x_test[i]-self.mu_clstr[c])
			clstr_lable[i] = np.argmin(dist)
			
		return 	clstr_lable

# ===============================================================
# kNN (binomial):
# ===============================================================
class kNN():
	def __init__(self, k=3):
		self.k = k

	def get_classes(self, lables):
		n = len(lables)
		self.classes = np.unique(lables)

	def fit(self, x_train, y_train):
		self.get_classes(y_train)
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		n = len(x_test)
		pred = np.zeros(n)
		m = len(self.x_train)
		
		for i in range(n):
			dist = []
			for j in range(m):
				d =np.linalg.norm(self.x_train[j]-x_test[i])
				dist.append([d, self.y_train[j]])
			
			min_dist = [i[1] for i in sorted(dist)]
			
			C1 = 0
			C2 = 0
			for k in range(self.k):	
								
				if min_dist[k] == self.classes[0]:
					C1 += 1					
				else:
					C2 += 1		

			if C1 > C2:
				pred[i] = self.classes[0]
			else:
				pred[i] = self.classes[1]

		return pred

	def score(self, x_test, y_test):
		y_pred = self.predict(x_test)
		correct = np.equal(y_pred, y_test)
		accuracy = np.sum(correct)/len(y_pred)
		
		return accuracy

# ===============================================================
# Linear Logistic Regression (binomial):
# ===============================================================

class LinearLogistic():
	def __init__(self, normalized_data=True, learning_rate=0.1, 
				fixed_learning=False, early_termination=False,
				epochs=50, batch_size=100, tol=1e-3):
		self.eta = learning_rate
		self.normalized_data = normalized_data
		self.fixed_learning = fixed_learning
		self.early_termination = early_termination
		self.epochs = epochs
		self.batch_size = batch_size
		self.tol = tol

	def __enter__(self):
		pass

	def __exit__(self):
		pass

	def error_message(self, msg):
		self.error = msg

	def weight_initialize(self, n):
		self.w = np.zeros(n+1)

	def normalize(self, xx):
		x = copy.deepcopy(xx)
		self.mu_x = np.mean(x, axis=0)
		self.std_x = np.std(x, axis=0)		
		self.std_x[self.std_x==0] = 1
		x = (x - self.mu_x) / self.std_x

		return x

	def subset_next_iter(self, x, y, step):
		n1 = step*self.batch_size
		n2 = (step+1)*self.batch_size

		xt = x[n1:n2]
		yt = y[n1:n2]		

		return xt, yt

	def reclassify(self, yy):
		y = copy.deepcopy(yy)	
		lables = np.unique(y)
		if len(lables) > 2:
			self.error_message("number of classes in the training data sets are greater than two")

		self.original_lable = {1:lables[0], -1:lables[1]}
		y[y==lables[0]] = -909 
		y[y==lables[1]] = -1
		y[y==-909] = 1

		return y

	def declassify(self, pred):
		pred[pred >= 0] = self.original_lable[1]
		pred[pred < 0] = self.original_lable[-1]
		return pred

	def activation(self, v):
		b = 10
		y = np.tanh(b*v)		
		return y

	def d_activation(self, v):
		b = 10		
		y = b*(1+np.tanh(b*v)**2)				
		return y

	def eta_annealing(self):
		pass


	def fit(self, xtrain, ytrain):
		if self.normalized_data is False:
			x = self.normalize(xtrain)
		else:
			x = xtrain

		y = self.reclassify(ytrain)  # +1 & -1

		n = x.shape		
		clmn_bias = np.ones((n[0],1), dtype=x.dtype) 
		x = np.append(x, clmn_bias, axis=1)
		
		self.weight_initialize(n[1])

		iteration = int(n[0]/self.batch_size)
		dw_mean = 1
		epoch = 0
		while epoch < self.epochs or dw_mean > self.tol:
			indices = [i for i in range(n[0])]
			random.shuffle(indices)

			x = x[indices]
			y = y[indices]

			for step in range(iteration):
				xt, yt = self.subset_next_iter(x, y, step)

				# regression layer: pred = activation(v = w.x)
				for i in range(self.batch_size):
					v = np.dot(self.w, xt[i])
					
								
					pred = self.activation(v)
								
					e = yt[i] - pred # error vector
															
					# minimizing cost by gradient descent:
					dw = -e*self.d_activation(v)*xt[i]
					self.w -= self.eta*dw
					
					dw_mean = np.mean(dw)
					if dw_mean < self.tol:
						break
						
				
				if dw_mean < self.tol:
					break

			epoch += 1

	def predict(self, xs):
		if self.normalized_data is False:
			xs = (xs - self.mu_x) / self.std_x

		n = len(xs)
		clmn_bias = np.ones((n,1), dtype=xs.dtype) 
		xs = np.append(xs, clmn_bias, axis=1)

		# logistic regression machine:
		v = np.dot(self.w, np.transpose(xs))
		pred = self.activation(v)

		pred = self.declassify(pred)

		return pred

	def score(self, xs, ys):
		y_pred1 = self.predict(xs)
		correct1 = np.equal(y_pred1, ys)
		accuracy1 = np.sum(correct1)/len(y_pred1)
		
		return accuracy1

# ===============================================================
# MLP (binomial):
# ===============================================================

class MLP():
	def __init__(self, shape=[5,1], normalized_data=True, learning_rate=0.01, 
				fixed_learning=False, early_termination=False,
				epochs=50, batch_size=100, tol=1e-3):
		self.eta = learning_rate
		self.normalized_data = normalized_data
		self.fixed_learning = fixed_learning
		self.early_termination = early_termination
		self.epochs = epochs
		self.batch_size = batch_size
		self.tol = tol
		self.shape = shape
		self.n_layers = len(shape)

	def __enter__(self):
		pass

	def __exit__(self):
		pass

	def error_message(self, msg):
		self.error = msg
	
	def normalize(self, xx):
		x = copy.deepcopy(xx)
		self.mu_x = np.mean(x, axis=0)
		self.std_x = np.std(x, axis=0)		
		self.std_x[self.std_x==0] = 1
		x = (x - self.mu_x) / self.std_x

		return x

	def subset_next_iter(self, x, y, step):
		n1 = step*self.batch_size
		n2 = (step+1)*self.batch_size

		xt = x[n1:n2]
		yt = y[n1:n2]		

		return xt, yt

	def reclassify(self, yy):
		y = copy.deepcopy(yy)	
		lables = np.unique(y)
		if len(lables) > 2:
			self.error_message("number of classes in the training data sets are greater than two")

		self.original_lable = {1:lables[0], -1:lables[1]}
		y[y==lables[0]] = -909 
		y[y==lables[1]] = -1
		y[y==-909] = 1

		return y

	def declassify(self, pred):
		pred[pred >= 0] = self.original_lable[1]
		pred[pred < 0] = self.original_lable[-1]
		return pred

	def activation(self, v):
		b = 10
		y = np.tanh(b*v)		
		return y

	def d_activation(self, v):
		b = 10
		y = b*(1+np.tanh(b*v)**2)				
		return y

	def eta_annealing(self):
		pass

	def weight_initialize(self, n):
		
		self.weights = {}
		self.biases = {}
		self.dw = {}
		self.db = {}
		
		for l in range(self.n_layers):

			if l==0:
				self.weights[l] = np.random.rand(self.shape[l],n)
				self.dw[l] = np.zeros((self.shape[l],n))
			else:
				self.weights[l] = np.random.rand(self.shape[l],self.shape[l-1])
				self.dw[l] = np.zeros((self.shape[l],self.shape[l-1]))
			
			self.biases[l] = np.random.rand(self.shape[l])			
			self.db[l] = np.zeros(self.shape[l])

	def architecture(self, n):
		# MLP architecture:
		self.y_layers = {}
		self.v_layers = {}
		self.delta_layers = {}

		self.y_layers[-1] = np.zeros((1,n))
		self.delta_layers[-1] = np.zeros((1,n))
			
		for l in range(self.n_layers):
			self.v_layers[l] = np.zeros((1,self.shape[l]))
			self.y_layers[l] = np.zeros((1,self.shape[l]))
			self.delta_layers[l] = np.zeros((1,self.shape[l]))

	def fit(self, xtrain, ytrain):
		if self.normalized_data is False:
			x = self.normalize(xtrain)
		else:
			x = xtrain

		y = self.reclassify(ytrain)  # +1 & -1

		n = x.shape		
			
		self.weight_initialize(n[1])

		self.architecture(n[1])

		iteration = int(n[0]/self.batch_size)
		dw_mean = 1
		epoch = 0
		

		while epoch < self.epochs:
			indices = [i for i in range(n[0])]
			random.shuffle(indices)

			x = x[indices]
			y = y[indices]

			for step in range(iteration):
				xt, yt = self.subset_next_iter(x, y, step)

				# regression layer: pred = activation(v = w.x)
				for i in range(self.batch_size):

					# go through layers:
					self.y_layers[-1] = xt[i]
					
					for l in range(self.n_layers):
						self.v_layers[l] = np.dot(self.weights[l], np.transpose(self.y_layers[l-1])) + self.biases[l]

						self.y_layers[l] = self.activation(self.v_layers[l])

					e = yt[i] - self.y_layers[self.n_layers-1] # error vector
							
					# back propogation:
					l = self.n_layers-1
					self.delta_layers[l] = e * self.d_activation(self.v_layers[l])
					for l in range(self.n_layers-2,-1,-1):
						self.delta_layers[l] = np.dot(np.transpose(self.weights[l+1]), self.delta_layers[l+1])
						self.delta_layers[l] *= self.d_activation(self.v_layers[l])

					# delta weights:
					for l in range(self.n_layers):							
						self.dw[l] = np.matmul(self.delta_layers[l].reshape(self.shape[l],1), self.y_layers[l-1].reshape(1,len(self.y_layers[l-1])))
						
						self.db[l] = self.delta_layers[l]

					# minimizing cost by gradient descent:					
					for l in range(self.n_layers):
						self.weights[l] += self.eta * self.dw[l]	
						self.biases[l] += self.eta * self.db[l]		
			
			epoch += 1
			

	def predict(self, xs):
		if self.normalized_data is False:
			xs = (xs - self.mu_x) / self.std_x

		n = len(xs)				

		# logistic regression machine:
		y_layers = {}
		v_layers = {}
		y_layers[-1] = np.transpose(xs)
		pred = np.zeros(len(xs))

		for i in range(len(xs)):
			self.y_layers[-1] = xs[i]
					
			for l in range(self.n_layers):
				self.v_layers[l] = np.dot(self.weights[l], np.transpose(self.y_layers[l-1])) + self.biases[l]

				self.y_layers[l] = self.activation(self.v_layers[l])

			pred[i] = self.y_layers[self.n_layers-1]

		return self.declassify(pred)

	def score(self, xs, ys):
		y_pred1 = self.predict(xs)
		correct1 = np.equal(y_pred1, ys)
		accuracy1 = np.sum(correct1)/len(y_pred1)
		
		return accuracy1


		