
#taken from CS231n Stanford Tutorial


import numpy as np
import matplotlib.pyplot as plt
import torch

#from classifiers.classification_cnn import ClassificationCNN
from src.data.DatasetGen import DatasetGen

#torch.set_default_tensor_type('torch.FloatTensor')
#set up default cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

# Load the (preprocessed) CIFAR10 data. The preprocessing includes
# channel swapping, normalization and train-val-test splitting.
# Loading the datasets might take a while.

datasetGen = DatasetGen()
datasetGen.BinaryShinyPokemonDataset()
data_dict = datasetGen.Subsample([0.8,0.1,0.1])
print("Train size: %i" % len(data_dict["X_train"]))
print("Val size: %i" % len(data_dict["X_val"]))
print("Test size: %i" % len(data_dict["X_test"]))


X = data_dict["X_train"]
y = data_dict["y_train"]
#Todo Get Vis working
classes = ['normal', 'shiny']
num_classes = len(classes)
samples_per_class = 7
for y_hat, cls in enumerate(classes):
    idxs = np.flatnonzero(y == y_hat)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y_hat + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X[idx].transpose(1,2,0).astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()