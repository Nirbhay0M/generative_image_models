import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

file = 'GEN_images/side-cnn.npy'
file2 = 'GEN_images/side-cnnnn.npy'
images = np.load(file)
nimages = np.load(file2)

# for i in range(len(images)):
# 	m = images[i].max()
# 	images[i] = np.where(images[i] > m-5, 1, 0)

# np.save(file, images)
t = 1
for i in range(len(images)):
	a = images[i].reshape((28, 28))
	# m = a.max()
	# a = np.where(a > m-5, 1, 0)
	plt.figure(0)
	plt.clf()
	plt.imshow(a, cmap='gray')
	plt.savefig("test/%d.png" %t)
	t += 1

	a = nimages[i].reshape((28, 28))
	plt.figure(0)
	plt.clf()
	plt.imshow(a, cmap='gray')
	plt.savefig("test/%d.png" %t)
	t+=1


# plotting mnist
# mnist_dir = 'data/mnist/'

# print ("Loading")
# mndata = MNIST(mnist_dir)
# tr_data = mndata.load_training()
# tr_data = np.asarray(tr_data[0])
# # tr_data = np.where(tr_data > 0, 1, 0)
# print tr_data.shape
# for i in range(len(tr_data)):
# 	a = tr_data[i].reshape((28,28))
# 	m = a.max()
# 	a = np.where(a > 0, 1, 0)
# 	plt.figure(0)
# 	plt.clf()
# 	plt.imshow(a, cmap='gray')
# 	plt.savefig("test/%d.png" %i)