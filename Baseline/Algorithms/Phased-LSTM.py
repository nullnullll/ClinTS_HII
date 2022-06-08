import torch
from torch import nn
import math
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
import pickle
import numpy as np
from sklearn import metrics
import Baseline.utils as utils

OFF_SLOPE=1e-3
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--async_sample', action='store_true', default=True,
                    help='Sample waves asynchronously')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--task', type=str, default='in_hospital_mortality')

args = parser.parse_args()


class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


# function to extract grad
def set_grad(var):
    def hook(grad):
        var.grad = grad

    return hook


class GradMod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, other):

        result = torch.fmod(input, other)
        ctx.save_for_backward(input, other)
        return result

    @staticmethod
    def backward(ctx, grad_output):

        x, y = ctx.saved_variables
        return grad_output * 1, grad_output * torch.neg(torch.floor_divide(x, y))


class PLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.Periods = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.Shifts = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.On_End = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        # Phased LSTM
        # -----------------------------------------------------
        nn.init.constant_(self.On_End, 0.05)  # Set to be 5% "open"
        nn.init.uniform_(self.Shifts, 0, 100)  # Have a wide spread of shifts
        # Uniformly distribute periods in log space between exp(1, 3)
        self.Periods.data.copy_(torch.exp((3 - 1) *
                                          torch.rand(self.Periods.shape) + 1))
        # -----------------------------------------------------

    def forward(self, x, ts,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        shift_broadcast = self.Shifts.view(1, -1)
        period_broadcast = abs(self.Periods.view(1, -1))
        on_mid_broadcast = abs(self.On_End.view(1, -1)) * 0.5 * period_broadcast
        on_end_broadcast = abs(self.On_End.view(1, -1)) * period_broadcast

        def calc_time_gate(time_input_n):
            # Broadcast the time across all units
            t_broadcast = time_input_n.unsqueeze(-1)
            # Get the time within the period
            in_cycle_time = GradMod.apply(t_broadcast + shift_broadcast, period_broadcast)

            # Find the phase
            is_up_phase = torch.le(in_cycle_time, on_mid_broadcast)
            is_down_phase = torch.gt(in_cycle_time, on_mid_broadcast) * torch.le(in_cycle_time, on_end_broadcast)

            # Set the mask
            sleep_wake_mask = torch.where(is_up_phase, in_cycle_time / on_mid_broadcast,
                                          torch.where(is_down_phase,
                                                      (on_end_broadcast - in_cycle_time) / on_mid_broadcast,
                                                      OFF_SLOPE * (in_cycle_time / period_broadcast)))
            return sleep_wake_mask

        # -----------------------------------------------------

        HS = self.hidden_size
        for t in range(seq_sz):
            old_c_t = c_t
            old_h_t = h_t
            x_t = x[:, t, :]
            t_t = ts[:, t]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            sleep_wake_mask = calc_time_gate(t_t)
            # Sleep if off, otherwise stay a bit on
            c_t = sleep_wake_mask * c_t + (1. - sleep_wake_mask) * old_c_t
            h_t = sleep_wake_mask * h_t + (1. - sleep_wake_mask) * old_h_t
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class Net(nn.Module):

    def __init__(self, inp_dim=23, hidden_dim=128, use_lstm=True):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
        if args.task == 'cip':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 4),
                torch.nn.Sigmoid())
        elif args.task == 'wbm':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 12),
                torch.nn.Sigmoid())
        elif args.task == 'los':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 9),
                torch.nn.Sigmoid())
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 2),
                torch.nn.Sigmoid())

        if use_lstm:
            pass
            # One extra vector for time
            self.rnn = CustomLSTM(inp_dim + 1, hidden_dim)
        else:
            self.rnn = PLSTM(inp_dim, hidden_dim)


    def forward(self, points, times):
        if self.use_lstm:
            combined_input = torch.cat((points, torch.unsqueeze(times, dim=-1)), -1)
            lstm_out, _ = self.rnn(combined_input)
        else:
            lstm_out, _ = self.rnn(points, times)
        classes = self.classifier(lstm_out)
        return classes


def train(args, model, device, train_loader, optimizer, epoch, dim=23, clip_value=100):
    # train the model
    if args.task == 'wbm':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    model.train()

    for train_batch, label in train_loader:
        train_batch, label = train_batch.to(device), label.to(device)
        batch_len = train_batch.shape[0]
        observed_data, observed_mask, observed_tp = \
            train_batch[:, :, :dim], train_batch[:, :, dim:2 * dim], train_batch[:, :, -1]

        optimizer.zero_grad()
        output = model(observed_data, observed_tp)

        loss = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()



def test(model, device, test_loader, criterion,dim=23):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred = []
    true = []
    with torch.no_grad():
        for test_batch, label in test_loader:
            test_batch, label = test_batch.to(device), label.to(device)
            batch_len = test_batch.shape[0]
            observed_data, observed_mask, observed_tp = \
                test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]
            output = model(observed_data, observed_tp)
            test_loss += criterion(output, label).item()
            pred.append(output.cpu().numpy())
            true.append(label.cpu().numpy())
    test_loss /= total

    pred = np.concatenate(pred, 0)
    pred = np.nan_to_num(pred)
    true = np.concatenate(true, 0)
    acc = np.mean(pred.argmax(1) == true)
    auroc = metrics.roc_auc_score(
        true, pred[:, 1])

    precision, recall, threshold = metrics.precision_recall_curve(true, pred[:, 1])
    auprc = metrics.auc(recall, precision)

    return test_loss / pred.shape[0], acc, auroc, auprc

def main():

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.task == 'in_hospital_mortality':
        data_obj = utils.get_mimiciii_data(args)
    elif args.task == 'decom':
        data_obj = utils.get_decom_data(args)
    elif args.task == 'cip':
        data_obj = utils.get_cip_data(args)
    elif args.task == 'wbm':
        data_obj = utils.get_wbm_data(args)
    elif args.task == 'los':
        data_obj = utils.get_los_data(args)

    torch.manual_seed(args.seed)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]

    print('Training Phased LSTM.')
    model = Net(use_lstm=False).to(device)
    if args.task == 'wbm':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val_auroc = float(0)
    if args.save_model:
        torch.save(model.state_dict(), "plstm_0.pt")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        val_loss,val_acc,val_auroc,val_auprc = test(model, device, val_loader,criterion)
        test_loss,test_acc,test_auroc,test_auprc = test(model, device, test_loader,criterion)
        message ='Iter: {},   val_auroc_macro: {:.4f}, val_auprc: {:.4f}, test_auroc_macro: {:.4f}, test_auprc: {:.4f}' \
            .format(epoch, val_auroc, val_auprc, test_auroc, test_auprc)
        print(message)
    if val_auroc >= best_val_auroc:

        best_val_auroc = max(best_val_auroc, val_auroc)
        best_test_auroc = test_auroc
        best_test_auprc = test_auprc
        model_state_dict = model.state_dict()


if __name__ == '__main__':
    main()