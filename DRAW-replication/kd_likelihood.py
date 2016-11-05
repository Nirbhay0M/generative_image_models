#!/usr/bin/env python

import os
import pickle
import numpy as np

from tensorflow.examples.tutorials import mnist
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

# MNIST Dataset
mnist_directory = os.path.join("./tmp", "mnist")
train_data = mnist.input_data.read_data_sets(mnist_directory, one_hot=True).train # binarized (0-1) mnist train data
test_data = mnist.input_data.read_data_sets(mnist_directory, one_hot=True).test # binarized (0-1) mnist test data

X = train_data.images #[1:10000]
y = train_data.labels

X_test = test_data.images
y_test = test_data.labels

params = {
	'kernel':['gaussian'],
	'bandwidth': np.logspace(-1.5, 0.5, 10)
}
# print "Performing GridSearchCV."
# grid = GridSearchCV(KernelDensity(), params)
# grid.fit(X)
# print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
# kde = grid.best_estimator_

# kde = KernelDensity(kernel='gaussian',bandwidth=0.25)
# kde.fit(X)
# print "Scoring test samples."
# kde.score_samples(X_test[1:100])

GMM_SAVE = "tmp/GMM_MODEL.tmp"

if os.path.isfile(GMM_SAVE):
	print "Loading GMM save from:%s"%GMM_SAVE
	gmm = pickle.load(GMM_SAVE)
else:
	gmm = GaussianMixture(10,random_state=42)
	print "Fitting GMM Model..."
	gmm.fit(X)
	with open(GMM_SAVE,"w") as fp:
		pickle.dump(gmm,fp)
		print "Saving GMM to:%s"%GMM_SAVE

