import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.autograd import Variable

from src.classifiers.classification_cnn import ClassificationCNN
from src.data.DatasetGen import DatasetGen, PytorchDataset, ConvertDatasetDictToTorch

def main():
    #torch.set_default_tensor_type('torch.FloatTensor')
    #set up default cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the (preprocessed) CIFAR10 data. The preprocessing includes
    # channel swapping, normalization and train-val-test splitting.
    # Loading the datasets might take a while.


    datasetGen = DatasetGen()
    datasetGen.BinaryShinyPokemonDataset()
    data_dict = datasetGen.Subsample([0.6,0.2,0.2])
    print("Train size: %i" % len(data_dict["X_train"]))
    print("Val size: %i" % len(data_dict["X_val"]))
    print("Test size: %i" % len(data_dict["X_test"]))

    train_data, val_data, test_data = ConvertDatasetDictToTorch(data_dict)

    from src.solver import Solver
    from torch.utils.data.sampler import SequentialSampler

    num_train = len(train_data)
    OverfitSampler = SequentialSampler(range(num_train))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)

    ############################################################################
    # TODO: Initialize a model and train it using your Solver class. Start     #
    # with the previously given set of hyperparameters.                        #
    ############################################################################

    model = ClassificationCNN(input_dim=[3,96,96],num_classes=2,
        num_filters=6, kernel_size=3, stride_conv=1,
        weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100, dropout=0.0)
    model.to(device)
    solver = Solver(optim_args={"lr": 1e-3, "weight_decay": 1e-3})
    solver.train(model, train_loader, val_loader, log_nth=10, num_epochs=5)

    from src.vis_utils import visualize_grid

    # first (next) parameter should be convolutional
    conv_params = next(model.parameters()).cpu().data.numpy()
    grid = visualize_grid(conv_params.transpose(0, 2, 3, 1))
    plt.imshow(grid.astype('uint8'))
    plt.axis('off')
    plt.gcf().set_size_inches(6, 6)
    plt.show()


if __name__ == '__main__':
    main()