import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

file = 'kernel/side-conv.npy'

filters = np.load(file)


# print filters[1][0].shape
filters = filters[0]

n = filters.shape[3]
h = filters.shape[0]
w = filters.shape[1]
im = np.zeros((h*2+4, w*8+16))
# im = np.where(im, 10000, 0)
print h,w,n
i = 0
j = 0
for t in range(n):
	a = filters[...,t]
	a = a[...,0]
	print a.shape
	im[i:i+h, j:j+w] = a
	i = (i+h+2)
	if(i == 2*h+4):
		i = 0
	if(i == 0):
		j=j+w+2
plt.figure(0)
plt.clf()
plt.matshow(im, cmap='gray')
plt.savefig("kernel/sfinal.png")
