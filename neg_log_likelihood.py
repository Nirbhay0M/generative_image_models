import numpy as np
import sys
from mnist import MNIST
from math import log

def entropy (gen, nn):
	nll = 0
	for i in range(len(gen)):
		nll += log(gen[i])*nn[i] + log((1-gen[i]))*(1-nn[i])
	return -nll

file_name = sys.argv[1]
# gen_images = np.load(file_name)
ts_data = mndata.load_testing()
gen_images = ts_data[0]
gen_images = np.asarray(gen_images)

n_images = len(gen_images)

nn_images = np.load(file_name[:-4]+'nn.npy')

nll = np.zeros(n_images)

for i in range(n_images):
	nll[i] = entropy(gen_images[i], nn_images[i])

print(file_name, nll.max, nll.min, np.average(nll))