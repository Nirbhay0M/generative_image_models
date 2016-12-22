import numpy as np
import sys
from mnist import MNIST
from math import log

epsilon = 0.000000001
def entropy (gen, nn):
	nll = 0
	for i in range(len(gen)):
		nll += log(gen[i]+epsilon)*nn[i] + log(1-gen[i]+epsilon)*(1-nn[i])
	return -nll

file_name = sys.argv[1]

gen_images = np.load(file_name)
gen_images = gen_images[0:1000]
# mnist_dir = 'data/mnist/'
# mndata = MNIST(mnist_dir)
# ts_data = mndata.load_testing()
# gen_images = ts_data[0][0:1000]
# gen_images = np.asarray(gen_images)
# gen_images = np.where(gen_images > 0, 1, 0)

n_images = len(gen_images)

nn_images = np.load(file_name[:-4]+'nn.npy')

nll = np.zeros(n_images)

for i in range(n_images):
	nll[i] = entropy(gen_images[i], nn_images[i])

print("file: %s\nmax error: %f\nmin error: %f\naverage error: %f\nstd: %f\n" %(file_name, np.max(nll), np.min(nll), np.average(nll), np.std(nll)))
