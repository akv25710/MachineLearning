# K fold cross validation ( uses multiple training and testing sets )

# import iris data set
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()

# first apply train test model with train 60% and test 40% insted of normal 20% test and 80% train model
# Split the iris data into train/test data sets with 40% reserved for testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# Build an SVC model for predicting iris classifications using training data
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

# Now measure its performance with the test data
clf.score(X_test, y_test)   
# 0.96666666666666667

# now using same iris data on the k flod cross validation
# We give cross_val_score a model, the entire data set and its "real" values, and the number of folds:
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)

# Print the accuracy for each fold:
print scores

# And the mean accuracy of all 5 folds:
print scores.mean()
# [ 0.96666667  1.          0.96666667  0.96666667  1.        ]
# mean of all the cross validation output
# 0.98

# cross validation using polynomial fitting
clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
print scores
print scores.mean()
# [ 1.          1.          0.9         0.93333333  1.        ]
# 0.966666666667

# svm using polynomial fit
# Build an SVC model for predicting iris classifications using training data
clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)

# Now measure its performance with the test data
clf.score(X_test, y_test)
# 0.96666666666666667

