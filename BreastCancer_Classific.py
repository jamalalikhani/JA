import pandas as pd 
import numpy as np 
from sklearn import preprocessing, neighbors, svm, cluster, linear_model
from sklearn.cross_validation import train_test_split
import copy 
from sklearn import ensemble, tree, neural_network, naive_bayes
import classification_methods as cm 
import random

data = pd.read_csv('breast-cancer-wisconsin.data.txt')
data.drop(['id'],axis=1,inplace=True)
data.replace('?',np.nan,inplace=True)
data.dropna(inplace=True)
data = data.astype(float).values.tolist()
data = np.array(data, dtype='float64')
random.shuffle(data)

x = data[:,:-1]
y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# ...................................................................................
clf1 = neighbors.KNeighborsClassifier(n_neighbors=5)
clf1.fit(x_train, y_train)
acc1 = clf1.score(x_test,y_test)
print("Accuracy by kNN: \t{0:.1%}".format(acc1))
y_pred = clf1.predict(x_test)
correct = np.equal(y_test, y_pred)
incorrect = (correct==False)

# ...................................................................................
clf = cm.kNN(k=5)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = clf.score(x_test, y_test)
print("Accuracy by kNN (scratch): {0:.1%}".format(acc))

# ...................................................................................
clf2 = svm.SVC()
clf2.fit(x_train,y_train)
acc2 = clf2.score(x_test,y_test)
print("Accuracy by SVM: {0:.1%}".format(acc2))

clf3 = svm.SVC()
x_train_scaled =preprocessing.scale(x_train)
x_test_scaled =preprocessing.scale(x_test)
clf3.fit(x_train_scaled,y_train)
acc3 = clf3.score(x_test_scaled,y_test)
print("Accuracy by SVM (with scaling): {0:.1%}".format(acc3))

# ...................................................................................
clf4 = cluster.KMeans(n_clusters=2)
clf4.fit(x_train)
centers = clf4.cluster_centers_
mean1 = np.mean(x_train[y_train==2],axis=0)
dist=np.zeros(2)
for i in range(len(centers)):
	dist[i] = np.linalg.norm(centers[i]-mean1)
c = np.argmin(dist)

y_test_r = copy.deepcopy(y_test)
y_test_r[y_test_r==2] = c
y_test_r[y_test_r==4] = 1-c
y_pred = clf4.predict(x_test)
correct4 = np.equal(y_pred,y_test_r)
acc4 = np.sum(correct4)/len(y_pred)
print("Accuracy by KMeans: {0:.1%}".format(acc4))

# ...................................................................................
clf5 = cluster.KMeans(n_clusters=2)
clf5.fit(x_train_scaled)
centers2 = clf5.cluster_centers_
mean1 = np.mean(x_train_scaled[y_train==2],axis=0)
dist=np.zeros(2)
for i in range(len(centers2)):
	dist[i] = np.linalg.norm(centers2[i]-mean1)
c = np.argmin(dist)

y_test_r = copy.deepcopy(y_test)
y_test_r[y_test_r==2] = c
y_test_r[y_test_r==4] = 1-c
y_pred = clf5.predict(x_test_scaled)
correct5 = np.equal(y_pred,y_test_r)
acc5 = np.sum(correct5)/len(y_pred)
print("Accuracy by KMeans (with scaling): {0:.1%}".format(acc5))

# ...................................................................................
clf15 = cm.KMeans(n_clusters=2)
clf15.fit(x_train)
mu_clstr = clf15.mu_clstr
y_pred = clf15.predict(x_test)

mean1 = np.mean(x_train[y_train==2],axis=0)
dist=np.zeros(2)
for i in range(len(mu_clstr)):
	dist[i] = np.linalg.norm(mu_clstr[i]-mean1)
c = np.argmin(dist)

y_test_r = copy.deepcopy(y_test)
y_test_r[y_test_r==2] = c
y_test_r[y_test_r==4] = 1-c

correct6 = np.equal(y_pred,y_test_r)
acc15 = np.sum(correct6)/len(y_pred)
print("Accuracy by KMeans (Scratch): {0:.1%}".format(acc15))

# ...................................................................................
clf6 = linear_model.LinearRegression()
clf6.fit(x_train,y_train)
acc6 = clf6.score(x_test,y_test)
print("Accuracy by LinearRegression: {0:.1%}".format(acc6))

# ...................................................................................
clf7 = linear_model.LinearRegression()
clf7.fit(x_train_scaled,y_train)
acc7 = clf7.score(x_test_scaled,y_test)
print("Accuracy by LinearRegression (with scaling): {0:.1%}".format(acc7))

# ...................................................................................
clf8 = linear_model.LogisticRegression(penalty='l2', C=1.0)
clf8.fit(x_train,y_train)
acc8 = clf8.score(x_test,y_test)
print("Accuracy by LogisticRegression: {0:.1%}".format(acc8))

# ...................................................................................
clf9 = linear_model.LogisticRegression(penalty='l1', C=0.1)
clf9.fit(x_train_scaled,y_train)
acc9 = clf9.score(x_test_scaled,y_test)
print("Accuracy by LogisticRegression (scaled): {0:.1%}".format(acc9))

# ...................................................................................
clf = cm.LinearLogistic(normalized_data=False,learning_rate=0.001, epochs=50, batch_size=100, tol=1e-5)
clf.fit(x_train, y_train)
acc16 = clf.score(x_test, y_test)
print("Accuracy by LinearLogistic (scratch): {0:.1%}".format(acc16))

# ...................................................................................
clf = cm.MLP(shape=[2,1], normalized_data=False,learning_rate=0.001, epochs=50, batch_size=100, tol=1e-5)
clf.fit(x_train, y_train)
acc = clf.score(x_test, y_test)
print("Accuracy by MLP (scratch): {0:.1%}".format(acc))

# ...................................................................................
clf10 = ensemble.RandomForestClassifier()
clf10.fit(x_train,y_train)
acc10 = clf10.score(x_test,y_test)
print("Accuracy by RandomForest: {0:.1%}".format(acc10))

# ...................................................................................
clf11 = ensemble.AdaBoostClassifier()
clf11.fit(x_train,y_train)
acc11 = clf11.score(x_test,y_test)
print("Accuracy by AdaBoost: {0:.1%}".format(acc11))

# ...................................................................................
clf12 = tree.DecisionTreeClassifier()
clf12.fit(x_train,y_train)
acc12 = clf12.score(x_test,y_test)
print("Accuracy by DecisionTree: {0:.1%}".format(acc12))

# ...................................................................................
clf13 = naive_bayes.GaussianNB()
clf13.fit(x_train,y_train)
acc13 = clf13.score(x_test,y_test)
print("Accuracy by Naive Bayes {0:.1%}".format(acc13))

# ...................................................................................
clf14 = cm.NaiveBayes()
clf14.fit(x_train,y_train)
y_pred = clf14.predict(x_test)
acc14 = clf14.score(x_test,y_test)
print("Accuracy by Naive Bayes (from scratch) {0:.1%}".format(acc14))




