from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Solver(object):
    default_adam_args = {"lr": 1e-3,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=1, log_nth=0, reg = 0.01, L1=False):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')

        model.zero_grad()
        for epoch in range(num_epochs):
            epoch_label = "Epoch 1/" + str(epoch + 1)
            for batch_idx, (data, target) in enumerate (train_loader):
                X = Variable(data.type(torch.FloatTensor))
                y = Variable(target.type(torch.LongTensor))
                iteration_label = "Iteration" + str(batch_idx + 1)
                #data, target = data.to(device), target.to(device)
                optim.zero_grad()
                output = model.forward(X)
                loss = self.loss_func(output, y)
                # L2 penalty is already implemented in optim and controlled with parameter weight decay
                if L1:
                    l1 = 0
                    for p in model.parameters():
                        l1 = l1 + p.abs().sum()
                    loss += reg * l1
                loss.backward()
                optim.step()
                self.train_loss_history.append(loss)
                if batch_idx % log_nth == log_nth - 1 :
                    print('Train batch with {} TRAIN loss: {}'.format(iteration_label, loss))
                if batch_idx == len(train_loader) - 1:
                    pred = output.max(1, keepdim=True)
                    train_correct, train_total = self.accuracy (output, y.type(torch.LongTensor))
                    train_accuracy = 100.0 * float(train_correct) / train_total
                    self.train_acc_history.append(train_accuracy)
                    print('{} TRAIN acc/loss: {}/{}'.format(epoch_label, train_accuracy, loss))

            accuracies = []
            losses = []
            for batch_idx, (data, target) in enumerate (val_loader):
                X = Variable(data.type(torch.FloatTensor))
                y = Variable(target.type(torch.LongTensor))
                output = model.forward(X)
                var_loss = self.loss_func(output, y)
                if L1:
                    l1 = 0
                    for p in model.parameters():
                        l1 = l1 + p.abs().sum()
                    var_loss += reg * l1
                val_correct, val_total = self.accuracy (output, y.type(torch.LongTensor))
                val_accuracy = 100.0 * float(val_correct) / val_total
                accuracies.append(val_accuracy)
                losses.append(var_loss)
            print('{} EVAL acc/loss: {}/{}'.format(epoch_label, np.mean(accuracies),var_loss))
            self.val_acc_history.append(np.mean(accuracies))
            self.val_loss_history.append(losses)

        print('FINISH.')
    
    def accuracy (self, output, y):
        total, correct = 0 , 0
        #calculate Accuracy
        _, predicted = torch.max(output.data,1)
        total += y.size(0)
        correct += (predicted == y).sum()
        return correct, total
