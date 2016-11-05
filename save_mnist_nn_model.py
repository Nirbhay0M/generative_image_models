import numpy as np
from mnist import MNIST
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib

mnist_dir = 'data/mnist/'

print ("Loading")
mndata = MNIST(mnist_dir)
tr_data = mndata.load_training()
tr_data = np.asarray(tr_data[0])
tr_data = np.where(tr_data > 0, 1, 0)

k = 1
d_metric = 'l2'

print ("Fitting")
neigh = NearestNeighbors(n_neighbors = k, metric = d_metric)
neigh.fit(tr_data)
joblib.dump(neigh, 'data/mnist_nn_model.pkl') 

# for loading
# neigh = joblib.load('data/mnist_nn_model.pkl') 