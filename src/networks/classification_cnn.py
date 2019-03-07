"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim, num_classes, convolutionalDims, hiddenDims, kernel_size=5,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2,
                 dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim
        self.convolutionalDims = convolutionalDims
        self.hiddenDims = hiddenDims

        # self.max_pool2d = nn.MaxPool2d(pool)
        # afterPoolSize = height // 2
        afterPoolSize = height
        for idx, conv in enumerate(convolutionalDims):
            if conv == 0:
                continue
            padding = (kernel_size) // 2
            setattr(self, "conv%d" % idx, nn.Conv2d(channels, conv, kernel_size, stride_conv, padding))
            channels = conv
            # getattr(self, "conv%d" % idx).weight.data.mul_(weight_scale)
            torch.nn.init.xavier_uniform_(getattr(self, "conv%d" % idx).weight)
            self.max_pool2d = nn.MaxPool2d(pool)
            afterPoolSize /= 2
        last_Dim = int(convolutionalDims[len(convolutionalDims) - 1] * afterPoolSize ** 2)
        for idx, hidden_dim in enumerate(hiddenDims):
            setattr(self, "fc%d" % idx, nn.Linear(last_Dim, int(hidden_dim), bias=True))
            torch.nn.init.xavier_uniform_(getattr(self, "fc%d" % idx).weight)
            last_Dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fclast = nn.Linear(hidden_dim, num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.fclast.weight)

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        # x = self.max_pool2d(x)
        for idx, conv in enumerate(self.convolutionalDims):
            x = getattr(self, "conv%d" % idx)(x)
            x = F.relu(x)
            x = self.max_pool2d(x)
        x = x.view(-1, self.num_flat_features(x))
        for idx, hidden_dim in enumerate(self.hiddenDims):
            x = F.relu(self.dropout(getattr(self, "fc%d" % idx)(x)))
        x = self.fclast(x)

        return x

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
