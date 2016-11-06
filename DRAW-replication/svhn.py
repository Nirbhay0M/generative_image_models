
import numpy as np
from scipy.io import loadmat
from skimage.color import rgb2gray

class SVHN(object):
	def __init__(self,grey=True,scale=True,one_hot=False):
		self.trainMatFile = "svhn_cropped/train_32x32.mat"
		self.testMatFile = "svhn_cropped/test_32x32.mat"

		self.X_train,self.X_test = None,None
		self.y_train,self.y_test = None,None

	def preprocess(self,MatFile,grey=True,scale=True,one_hot=False):
		data = loadmat(MatFile)
		X = np.transpose(data['X'],(3,0,1,2))
		y = np.reshape(data['y'],(-1))

		if grey:
			X = self._grey(X)

		if not grey and scale:
			X = X/255.0

		X = np.reshape(X,[-1,1024])
		return X,y

	def _grey(self,X):
		X_list = map(rgb2gray,X)
		return np.array(X_list)

	def train(self):
		if self.X_train is None:
			self.X_train,self.y_train = self.preprocess(self.trainMatFile)
		return Dataset(self.X_train,self.y_train)

	def test(self):
		if self.X_test is None:
			self.X_test,self.y_test = self.preprocess(self.testMatFile)
		return Dataset(self.X_test,self.y_test)

class Dataset(object):
	def __init__(self,X,y):
		self._X = X
		self._y = y
		self._i = 0
		self._max = len(y)
		self._num_epochs = 0

	@property
	def X(self):
		return self._X

	@property
	def y(self):
		return self._y

	def next_batch(self,batch_size):
		if self._i+batch_size < self._max:
			self._i += batch_size
			return self._X[self._i:self._i+batch_size],\
					self._y[self._i:self._i+batch_size]
		else:
			# dif = self._max - self._i
			# X = np.concatenate(self._X[self._i:])
			self._i = 0
			self._num_epochs += 1

if __name__ == '__main__':
	a = SVHN()
	X = a.test()
