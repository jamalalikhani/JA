import pandas as pd 
import numpy as np 
import tensorflow as tf 
import random
import matplotlib.pyplot as plt


Bset = pd.read_csv('DataSetBTrain_clean.csv')
Bset = np.array(Bset,dtype='int')
print(Bset)
N = Bset.shape
test_size = int(0.1*N[0])
random.shuffle(Bset)
train_set = Bset[:-test_size]
test_set = Bset[:-test_size]

def creat_batch(Train_data,n_classes,batch_size=100,one_hot=True):
	random.shuffle(Train_data)
	X_train = Train_data[:batch_size,:-1]
	Y_train = Train_data[:batch_size,-1:]
	
	if one_hot==True:
		Y_train_one_hot = []
		for y in Y_train:
			class_out = [0, 0]
			if y[0]==2:
				class_out[0]+=1
			else:
				class_out[1]+=1
			Y_train_one_hot.append(class_out)
		return X_train, Y_train_one_hot
	else:
		return X_train, Y_train

n_h1 = 10
n_classes = 2
dimension = N[1]-1

x = tf.placeholder(tf.float32,[None,dimension])
y_true = tf.placeholder(tf.float32,[None,n_classes])


hl_1 = {'w':tf.Variable(tf.random_normal([dimension,n_h1])),'b':tf.Variable(tf.random_normal([n_h1]))}
v1 = tf.add(tf.matmul(x,hl_1['w']),hl_1['b'])
y1 = tf.nn.relu(v1)

outl = {'w':tf.Variable(tf.random_normal([n_h1,n_classes])),'b':tf.Variable(tf.random_normal([n_classes]))}
prediction = tf.add(tf.matmul(y1,outl['w']),outl['b'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y_true))
optimizer = tf.train.AdamOptimizer().minimize(cost)
correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct,'float'))

#run the graphs:
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	batch_size = 100
	n_epochs = 50
	loss_trend = np.zeros(n_epochs)

	for n in range(n_epochs):
		epoch_loss = 0
		for _ in range(int(len(train_set)/batch_size)):
			X_train, Y_train = creat_batch(train_set,2,batch_size,one_hot=True)
			_,c = sess.run([optimizer, cost],feed_dict={x:X_train,y_true:Y_train})
			epoch_loss += c
		loss_trend[n] = epoch_loss
		print('epoch = ',n,' loss func = ', epoch_loss)
	
	x_loos_trend = [i+1 for i in range(n_epochs)]
	plt.plot(x_loos_trend,loss_trend)
	plt.ylabel('Loss function value')
	plt.xlabel('epochs')
	plt.show()
	plt.close()
	
	X_test, Y_test = creat_batch(test_set,2,len(test_set),one_hot=True)
	test_accuracy = sess.run(accuracy,feed_dict={x:X_test,y_true:Y_test})
	print("accuracy: {0:.1%}".format(test_accuracy))

	w1 = sess.run(hl_1['w'])
	print("weight values: ")
	print(w1)





