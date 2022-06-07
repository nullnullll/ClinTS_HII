import argparse

import torch
import pickle
import numpy as np
import pandas as pd
import os
import math
import warnings
import itertools
import numbers
import torch.utils.data as utils
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--task', type=str, default='in_hospital_mortality')

args = parser.parse_args()


def get_mean(x):
    x_mean = []
    for i in range(x.shape[0]):
        mean = np.mean(x[i])
        x_mean.append(mean)
    return x_mean

def get_median(x):
    x_median = []
    for i in range(x.shape[0]):
        median = np.median(x[i])
        x_median.append(median)
    return x_median

def get_std(x):
    x_std = []
    for i in range(x.shape[0]):
        std = np.std(x[i])
        x_std.append(std)
    return x_std

def get_var(x):
    x_var = []
    for i in range(x.shape[0]):
        var = np.var(x[i])
        x_var.append(var)
    return x_var


def normalize_chk(dataset):
    all_x_add = np.zeros((dataset[0][0].shape[0], 1))
    for i in range(dataset.shape[0]):
        all_x_add = np.concatenate((all_x_add, dataset[i][0]), axis=1)

    mean = get_mean(all_x_add)
    median = get_median(all_x_add)
    std = get_std(all_x_add)
    var = get_var(all_x_add)

    print('mean')
    print(mean)
    print('median')
    print(median)
    print('std')
    print(std)
    print('var')
    print(var)

    return mean, median, std, var

def process_data(x, input_dim ):
    observed_vals, observed_mask, observed_tp = x[:, :,
                                                :input_dim], x[:, :, input_dim:2 * input_dim], x[:, :, -1]
    if np.max(observed_tp) != 0.:
        observed_tp[:] = observed_tp / np.max(observed_tp)
    if True:
        for k in range(input_dim):
            data_min, data_max = float('inf'), 0.
            for i in range(observed_vals.shape[0]):
                for j in range(observed_vals.shape[1]):
                    if observed_mask[i, j, k]:
                        data_min = min(data_min, observed_vals[i, j, k])
                        data_max = max(data_max, observed_vals[i, j, k])
            if data_max == 0:
                data_max = 1
            observed_vals[:, :, k] = ( observed_vals[:, :, k] - data_min) / data_max
    observed_vals[observed_mask == 0] = 0


class GRUD(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, x_mean=0, \
                 bias=True, batch_first=False, bidirectional=False, dropout_type='mloss', dropout=0):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.zeros = torch.autograd.Variable(torch.zeros(input_size))
        self.x_mean = torch.autograd.Variable(torch.FloatTensor(x_mean))
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if args.task == 'cip':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(23, 4),
                torch.nn.Sigmoid())
        elif args.task == 'wbm':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(23, 12),
                torch.nn.Sigmoid())
        elif args.task == 'los':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(23, 9),
                torch.nn.Sigmoid())
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(23, 2),
                torch.nn.Sigmoid())

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        self._all_weights = []

        '''
        w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
        w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
        b_ih = Parameter(torch.Tensor(gate_size))
        b_hh = Parameter(torch.Tensor(gate_size))
        layer_params = (w_ih, w_hh, b_ih, b_hh)
        '''
        # decay rates gamma
        w_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
        w_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        # z
        w_xz = torch.nn.Parameter(torch.Tensor(input_size))
        w_hz = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mz = torch.nn.Parameter(torch.Tensor(input_size))

        # r
        w_xr = torch.nn.Parameter(torch.Tensor(input_size))
        w_hr = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mr = torch.nn.Parameter(torch.Tensor(input_size))

        # h_tilde
        w_xh = torch.nn.Parameter(torch.Tensor(input_size))
        w_hh = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mh = torch.nn.Parameter(torch.Tensor(input_size))

        # y (output)
        w_hy = torch.nn.Parameter(torch.Tensor(32, hidden_size))

        # bias
        b_dg_x = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_y = torch.nn.Parameter(torch.Tensor(output_size))

        layer_params = (w_dg_x, w_dg_h, \
                        w_xz, w_hz, w_mz, \
                        w_xr, w_hr, w_mr, \
                        w_xh, w_hh, w_mh, \
                        w_hy, \
                        b_dg_x, b_dg_h, b_z, b_r, b_h, b_y)

        param_names = ['weight_dg_x', 'weight_dg_h', \
                       'weight_xz', 'weight_hz', 'weight_mz', \
                       'weight_xr', 'weight_hr', 'weight_mr', \
                       'weight_xh', 'weight_hh', 'weight_mh', \
                       'weight_hy']
        if bias:
            param_names += ['bias_dg_x', 'bias_dg_h', \
                            'bias_z', \
                            'bias_r', \
                            'bias_h', \
                            'bias_y']

        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """
        Resets parameter data pointer so that they can use faster code paths.
        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(GRUD, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(GRUD, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []

        weights = ['weight_dg_x', 'weight_dg_h', \
                   'weight_xz', 'weight_hz', 'weight_mz', \
                   'weight_xr', 'weight_hr', 'weight_mr', \
                   'weight_xh', 'weight_hh', 'weight_mh', \
                   'weight_hy', \
                   'bias_dg_x', 'bias_dg_h', \
                   'bias_z', 'bias_r', 'bias_h', 'bias_y']

        if self.bias:
            self._all_weights += [weights]
        else:
            self._all_weights += [weights[:2]]

    @property
    def _flat_weights(self):
        return list(self._parameters.values())

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def forward(self, input):
        # input.size = (3, 23,200) : num_input or num_hidden, num_layer or step
        X = torch.squeeze(input[:,0])  # .size = (23,200)
        Mask = torch.squeeze(input[:,1])  # .size = (23,200)
        Delta = torch.squeeze(input[:,2])  # .size = (23,200)
        Hidden_State = torch.autograd.Variable(torch.zeros(X.shape[0],self.input_size))

        step_size = X.size(1)  # 200
        # print('step size : ', step_size)

        output = None
        h = Hidden_State

        # decay rates gamma
        w_dg_x = getattr(self, 'weight_dg_x')
        w_dg_h = getattr(self, 'weight_dg_h')

        # z
        w_xz = getattr(self, 'weight_xz')
        w_hz = getattr(self, 'weight_hz')
        w_mz = getattr(self, 'weight_mz')

        # r
        w_xr = getattr(self, 'weight_xr')
        w_hr = getattr(self, 'weight_hr')
        w_mr = getattr(self, 'weight_mr')

        # h_tilde
        w_xh = getattr(self, 'weight_xh')
        w_hh = getattr(self, 'weight_hh')
        w_mh = getattr(self, 'weight_mh')

        # bias
        b_dg_x = getattr(self, 'bias_dg_x')
        b_dg_h = getattr(self, 'bias_dg_h')
        b_z = getattr(self, 'bias_z')
        b_r = getattr(self, 'bias_r')
        b_h = getattr(self, 'bias_h')

        for layer in range(num_layers):

            x = torch.squeeze(X[:, :, layer:layer + 1])
            m = torch.squeeze(Mask[:, :, layer:layer + 1])
            d = torch.squeeze(Delta[:, :, layer:layer + 1])

            # (4)
            # gamma_x = torch.exp(-torch.max(self.zeros, (w_dg_x * d + b_dg_x)))
            # gamma_h = torch.exp(-torch.max(self.zeros, (w_dg_h * d + b_dg_h)))

            # (5)
            # x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * self.x_mean)
            x = m * x + (1 - m) * self.x_mean


            # (6)
            if self.dropout == 0:
                # h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'Moon':
                '''
                RNNDROP: a novel dropout for rnn in asr(2015)
                '''
                # h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))

                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

            elif self.dropout_type == 'Gal':
                '''
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                '''
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

                # h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''

                # h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = dropout(h_tilde)

                h = (1 - z) * h + z * h_tilde

            else:
                # h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

        w_hy = getattr(self, 'weight_hy')
        b_y = getattr(self, 'bias_y')

        # output = torch.matmul(w_hy, h) + b_y
        # output = torch.sigmoid(output)
        output = self.classifier(h)
        return output


def fit(args, model, criterion, learning_rate, \
        train_dataloader, dev_dataloader, test_dataloader, \
        learning_rate_decay=0, n_epochs=30):
    epoch_losses = []
    best_val_auroc = float(0)
    # to check the update
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()

    for epoch in range(n_epochs):

        if learning_rate_decay != 0:

            # every [decay_step] epoch reduce the learning rate by half
            if epoch % learning_rate_decay == 0:
                learning_rate = learning_rate / 2
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))

        # train the model
        losses, acc = [], []
        label, pred = [], []
        y_pred_col = []
        model.train()
        for train_data, train_label in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Squeeze the data [1, 23, 200], [1,5] to [23, 200], [5]
            train_data = torch.squeeze(train_data)
            # train_label = torch.squeeze(train_label)

            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(train_data)

            pred.append(y_pred.cpu().detach().numpy())
            label.append(train_label.cpu().detach().numpy())

            # Compute loss
            loss = criterion(y_pred, train_label)

            losses.append(loss.item())

            # perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

        pred = np.concatenate(pred, 0)
        label = np.concatenate(label, 0)
        train_acc = np.mean(pred.argmax(1) == label)
        train_loss = np.mean(losses)

        train_pred_out = pred
        train_label_out = label

        # save new params
        new_state_dict = {}
        for key in model.state_dict():
            new_state_dict[key] = model.state_dict()[key].clone()

        # compare params
        # for key in old_state_dict:
        #     if (old_state_dict[key] == new_state_dict[key]).all():
        #         print('Not updated in {}'.format(key))

        # dev loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for dev_data, dev_label in dev_dataloader:
            # Squeeze the data [1, 23, 200], [1,5] to [23, 200], [5]
            dev_data = torch.squeeze(dev_data)
            # dev_label = torch.squeeze(dev_label)

            # Forward pass : Compute predicted y by passing train data to the model
            with torch.no_grad():
                y_pred = model(dev_data)

            # Save predict and label
            pred.append(y_pred.cpu().numpy())
            label.append(dev_label.cpu().numpy())

            # Compute loss
            loss = criterion(y_pred, dev_label)
            losses.append(loss.item())

        pred = np.concatenate(pred, 0)
        label = np.concatenate(label, 0)
        dev_acc =  np.mean(pred.argmax(1) == label)
        dev_loss = np.mean(losses)

        dev_pred_out = pred
        dev_label_out = label

        pred = np.asarray(pred)
        label = np.asarray(label)

        dev_auroc = roc_auc_score(label, pred[:, 1])

        precision, recall, threshold = metrics.precision_recall_curve(label, pred[:, 1])
        dev_auprc = metrics.auc(recall, precision)


        # test loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for test_data, test_label in test_dataloader:
            # Squeeze the data [1, 23, 200], [1,5] to [23, 200], [5]
            test_data = torch.squeeze(test_data)
            # test_label = torch.squeeze(test_label)

            # Forward pass : Compute predicted y by passing train data to the model
            with torch.no_grad():
                y_pred = model(test_data)

            # Save predict and label
            pred.append(y_pred.cpu().numpy())
            label.append(test_label.cpu().numpy())

            # Compute loss
            loss = criterion(y_pred, test_label)
            losses.append(loss.item())

        pred = np.concatenate(pred, 0)
        label = np.concatenate(label, 0)
        test_acc = np.mean(pred.argmax(1) == label)
        test_loss = np.mean(losses)

        test_pred_out = pred
        test_label_out = label

        epoch_losses.append([
            train_loss, dev_loss, test_loss,
            train_acc, dev_acc, test_acc,
            train_pred_out, dev_pred_out, test_pred_out,
            train_label_out, dev_label_out, test_label_out,
        ])

        pred = np.asarray(pred)
        label = np.asarray(label)

        test_auroc = roc_auc_score(label, pred[:, 1])

        precision, recall, threshold = metrics.precision_recall_curve(label, pred[:, 1])
        test_auprc = metrics.auc(recall, precision)

        # print("Epoch: {} Train: {:.4f}/{:.2f}%, Dev: {:.4f}/{:.2f}%, Test: {:.4f}/{:.2f}% AUC: {:.4f}".format(
        #     epoch, train_loss, train_acc*100, dev_loss, dev_acc*100, test_loss, test_acc*100, auc_score))
        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Dev auroc: {:.4f}, Dev auprc: {:.4f}, Test loss: {:.4f}, Test auroc: {:.4f}, Test auprc: {:.4f}".format(
            epoch, train_loss, dev_loss, dev_auroc, dev_auprc, test_loss, test_auroc, test_auprc ))

        if dev_auroc >= best_val_auroc:

            best_val_auroc = max(best_val_auroc, dev_auroc)
            best_test_auroc = test_auroc
            best_test_auprc = test_auprc
            model_state_dict = model.state_dict()

    return epoch_losses

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def data_dataloader(dataset, outcomes, \
                    train_proportion=0.8, dev_proportion=0.2, test_proportion=0.2):
    train_index = int(np.floor(dataset.shape[0] * train_proportion))
    dev_index = int(np.floor(dataset.shape[0] * (train_proportion - dev_proportion)))

    # split dataset to tarin/dev/test set
    train_data, train_label = dataset[:train_index, :, :, :], outcomes[:train_index]
    dev_data, dev_label = dataset[dev_index:train_index, :, :, :], outcomes[dev_index:train_index]
    test_data, test_label = dataset[train_index:, :, :, :], outcomes[train_index:]

    # ndarray to tensor
    train_data, train_label = torch.from_numpy(train_data).float(), torch.from_numpy(train_label).long()
    dev_data, dev_label = torch.from_numpy(dev_data).float(), torch.from_numpy(dev_label).long()
    test_data, test_label = torch.from_numpy(test_data).float(), torch.from_numpy(test_label).long()

    # tensor to dataset
    train_dataset = utils.TensorDataset(train_data, train_label)
    dev_dataset = utils.TensorDataset(dev_data, dev_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    # dataset to dataloader
    train_dataloader = utils.DataLoader(train_dataset,batch_size=32)
    dev_dataloader = utils.DataLoader(dev_dataset,batch_size=32)
    test_dataloader = utils.DataLoader(test_dataset,batch_size=32)

    print("train_data.shape : {}\t train_label.shape : {}".format(train_data.shape, train_label.shape))
    print("dev_data.shape : {}\t dev_label.shape : {}".format(dev_data.shape, dev_label.shape))
    print("test_data.shape : {}\t test_label.shape : {}".format(test_data.shape, test_label.shape))

    return train_dataloader, dev_dataloader, test_dataloader

if __name__ == '__main__':
    x = np.load('../Dataset/'+args.task+'/input.npy', allow_pickle=True)
    y = np.load('../Dataset/'+args.task+'/output.npy', allow_pickle=True)

    x = x[:, :47]
    y = y[:]

    input_dim = 23
    x = np.transpose(x, (0, 2, 1))
    process_data(x, input_dim)
    print(x.shape, y.shape)

    x = np.transpose(x, (0, 2, 1))
    x = torch.from_numpy(x).float()

    input_dim = 23
    observed_vals, observed_mask, observed_tp = x[:, :input_dim, :], x[:, input_dim:2 * input_dim, :], x[:, -1, :]

    delta = torch.zeros((observed_vals.shape[0], 23, 200))

    observed_vals = observed_vals[:, np.newaxis, :]
    observed_mask = observed_mask[:, np.newaxis, :]
    delta = delta[:, np.newaxis, :]
    x = np.concatenate((observed_vals, observed_mask, delta), axis=1)
    print(x.shape)
    print(y.shape)

    train_dataloader, dev_dataloader, test_dataloader = data_dataloader(x, y, train_proportion=0.8, dev_proportion=0.2)

    nor_mean, nor_median, nor_std, nor_var = normalize_chk(x)

    input_size = 23
    hidden_size = 23
    output_size = 1
    num_layers = 50

    x_mean = nor_mean
    x_median = nor_median

    # dropout_type : Moon, Gal, mloss
    model = GRUD(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=0,
                 dropout_type='mloss', x_mean=x_mean, num_layers=num_layers)


    count = count_parameters(model)
    print('number of parameters : ', count)
    print(list(model.parameters())[0].grad)

    if args.task == 'wbm':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    learning_rate = args.lr
    learning_rate_decay = 7
    n_epochs = args.epochs

    # learning_rate = 0.1 learning_rate_decay=True
    epoch_losses = fit(args, model, criterion, learning_rate, \
                       train_dataloader, dev_dataloader, test_dataloader, \
                       learning_rate_decay, n_epochs)