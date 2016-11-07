import numpy as np
import sys
from mnist import MNIST
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib

file_name = sys.argv[1]

mnist_dir = 'data/mnist/'

print ("Loading")
mndata = MNIST(mnist_dir)
tr_data = mndata.load_training()
tr_data = np.asarray(tr_data[0])
tr_data = np.where(tr_data > 0, 1, 0)
img_size = len(tr_data[0])

gen_images = np.load(file_name)
gen_images = gen_images[0:1000]
# ts_data = mndata.load_testing()
# gen_images = ts_data[0][0:1000]
# gen_images = np.asarray(gen_images)

gen_images = np.where(gen_images > 0, 1, 0)

n_images = len(gen_images)

k = 1
d_metric = 'l2'

# print ("Fitting")
# neigh = NearestNeighbors(n_neighbors = k, metric = d_metric)
# neigh.fit(tr_data)

# Load pretrained model
print("Loading saved model")
neigh = joblib.load('data/model/mnist_nn_model.pkl') 

print ("Predicting")
results = neigh.kneighbors(gen_images, k, return_distance = False)
results_images = np.zeros((n_images, img_size))

print ("Saving")
for i in range(n_images):
	results_images[i] = tr_data[results[i][0]]
np.save(file_name[:-4]+'nn', results_images)
