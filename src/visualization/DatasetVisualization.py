
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
data_dict = datasetGen.Subsample([0.6,0.2,0.2])
print("Train size: %i" % len(data_dict["X_train"]))
print("Val size: %i" % len(data_dict["X_val"]))
print("Test size: %i" % len(data_dict["X_test"]))


#Todo Get Vis working
classes = ['normal', 'shiny']
num_classes = len(classes)
samples_per_class = 7
for cls_idx, cls in enumerate(classes):
    cls_data = [datum for datum in testdata if datum[1] == cls_idx]
    rnd_idxs = np.random.randint(0, len(cls_data), samples_per_class)
    rnd_cls_data = [datum for i, datum in enumerate(cls_data) if i in rnd_idxs]
    for i, cls_datum in enumerate(rnd_cls_data):
        plt_idx = i * num_classes + cls_idx + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(cls_datum[0].numpy().transpose(1,2,0))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()