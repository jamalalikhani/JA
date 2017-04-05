# A Package of 10 classification algorithms on Breast Cancer dataset:
In this attempt, I have used `sklearn`, `TensorFlow`, and `my own implementation` to compare the performance and accuracy of different machine learning algorithms over Breast Cancer data set with binomia calsses and 9 dimensional features. 

You are more than welcome to use the codes. Please email me if you find something usefull to share at: jamal.alikhani@gmail.com. 

## Accuracy of Each Classification ALgorithm over a same data set:
You have to run `python3 BreastCancer_Classific.py` to call the classification algorithms from either the `sklearn` or from my own implementation (from scratch) of `classification_methods.py`.

Results are shown in below:
* Accuracy by kNN: 	                            98.5%
* Accuracy by kNN (scratch):                    97.8%
* Accuracy by SVM:                              99.3%
* Accuracy by SVM (with scaling):               97.1%
* Accuracy by KMeans:                           89.8%
* Accuracy by KMeans (with scaling):            96.4%
* Accuracy by KMeans (Scratch):                 89.8%
* Accuracy by LinearRegression:                 75.8%
* Accuracy by LinearRegression (with scaling):  76.1%
* Accuracy by LogisticRegression:               93.4%
* Accuracy by LogisticRegression (scaled):      97.1%
* Accuracy by LinearLogistic (scratch):         94.2%
* Accuracy by MLP (scratch):                    94.9%
* Accuracy by RandomForest:                     99.3%
* Accuracy by AdaBoost:                         99.3%
* Accuracy by DecisionTree:                     99.3%
* Accuracy by Naive Bayes                       95.6%
* Accuracy by Naive Bayes (from scratch)        95.6%

A sepearte code is also implemented to apply TensorFlow on the Breast Cancer dataset. Please run `BreastCancer_MLP_TenFlw.py`. The accuracy of 2 layers, each with 10 neurons is:
* Accuracy by MLP (TensorFolo):                 100.0%
