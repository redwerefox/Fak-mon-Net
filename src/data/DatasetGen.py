import os

import numpy as np
import torch
import torch.utils.data as data
import PIL
from PIL import Image
import os.path
import _pickle as pickle

class OverfitSampler(object):
    """
    Sample dataset to overfit.
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples

class PytorchDataset (data.Dataset):
	def __init__ (self, X, y):
		self.y = y
		self.X = X
		print("init set")

	def __getitem__(self, index):
		img = self.X[index]
		label = self.y[index]

		img = torch.from_numpy(img)
		return img, label

	def __len__(self):
		return len(self.y)


def ConvertDatasetDictToTorch(datasetDict):
	train_data, train_label = datasetDict["X_train"], datasetDict["y_train"]
	valid_data, valid_label = datasetDict["X_val"], datasetDict["y_val"]
	test_data, test_label = datasetDict["X_test"], datasetDict["y_test"]

	return ( PytorchDataset(train_data, train_label),
			PytorchDataset(valid_data, valid_label),
			PytorchDataset(test_data, test_label))

class DatasetGen():

	def BinaryShinyPokemonDataset(self, normalize=False):
		X = []
		y = []
		self.num_count = 0
		#num_total = num_training + num_validation + num_test

		#using Path for Windows/OS/Linux compability
		datasetFolder = ("../dataset/pokemon/main-sprites/black-white/")
		datasetFolderShiny = ("../dataset/pokemon/main-sprites/black-white/shiny/")

		#load not shiny pokemon
		for filename in os.listdir(datasetFolder):
			if filename.endswith("png"):
				image = Image.open(datasetFolder + filename)
				image = image.convert("RGB")
				#image = image.resize((dim,dim), PIL.Image.ANTIALIAS)
				image = np.array(image)
				#pytorch requires (channel x width x height)
				image = image.transpose(2,0,1)
				X.append(image)
				y.append(0) #not shiny
				self.num_count += 1

		#shiny
		for filename in os.listdir(datasetFolderShiny):
			if filename.endswith("png"):
				image = Image.open(datasetFolderShiny + filename)
				image = image.convert("RGB")
				# image = image.resize((dim,dim), PIL.Image.ANTIALIAS)
				image = np.array(image).transpose(2,0,1)
				X.append(image)
				y.append(1)
				self.num_count += 1


		randomize = np.arange(len(y))
		np.random.shuffle(randomize)

		X = np.array(X)
		y = np.array(y)

		X = X[randomize]
		y = y[randomize]

		self.X = X
		self.y = y

		print("Dataset shape after shuffle: " + str(X.shape))

		if(normalize):
			pass
			#X = np.divide(X, 255)
			#Normalize the data: subtract the mean image
			#mean_image = np.mean(X, axis=0)
			#X -= mean_image
			#todo variance divide



		dict = {"data" : X, "labels" : y }

		with open('shinyDataset.p', 'wb') as file:
			pickle.dump(dict, file)

		print("data saved in pickle")
		print("pictures have size x,y |" +	str(X.shape))

		return True
	

	def Subsample(self, datasetSplit = [0.6,0.2,0.2]):

		num_training = int(self.num_count * datasetSplit[0])
		num_validation = int(self.num_count * datasetSplit[1])
		num_test = int(self.num_count * datasetSplit[2])

		# Subsample the data
		mask = range(num_training)
		X_train = self.X[mask]
		y_train = self.y[mask]
		mask = range(num_training, num_training + num_validation)
		X_val = self.X[mask]
		y_val = self.y[mask]
		mask = range(num_training + num_validation,
					 num_training + num_validation + num_test)
		X_test = self.X[mask]
		y_test = self.y[mask]

		return {"X_train" : X_train, "y_train" : y_train, "X_val" : X_val, "y_val" : y_val, "X_test" : X_test, "y_test" : y_test}

	
def load_image( infilename ) :
	img = Image.open( infilename )
	img.load()
	data = np.asarray( img, dtype="int32" )
	return data

def save_image( npdata, outfilename ) :
	img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
	img.save( outfilename )

def rel_error(x, y):
	""" Returns relative error """
	assert x.shape == y.shape, "tensors do not have the same shape. %s != %s" % (x.shape, y.shape)
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def main():
	datasetGen = DatasetGen()
	datasetGen.BinaryShinyPokemonDataset()

if __name__ == "__main__":
	main()