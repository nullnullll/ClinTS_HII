# pylint: disable=E1101
import logging
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, args, device):
    observed_data, observed_mask \
        = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

    noise_std = args.std  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    num = observed_mask.sum(-1).sum(-1)
    num = torch.where(num == 0., torch.Tensor([1]).to(device), num.float())
    if args.norm:
        logpx /= num
        analytic_kl /= num
    return logpx, analytic_kl

def evaluate_classifier_cip(model, test_loader, dec=None, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    pred_scores = []
    labels_classes = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]

        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                       :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                out = classifier(z0)
                scores = torch.softmax(out, dim=1)
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                if args.task == 'cip':
                    label_classes = label_binarize(label.cpu().numpy(), classes=range(4))
                else:
                    label_classes = label_binarize(label.cpu().numpy(), classes=range(9))
                test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
        pred_scores.append(scores.cpu().numpy())
        labels_classes.append(label_classes)
    pred = np.concatenate(pred, 0)
    pred = np.nan_to_num(pred)
    true = np.concatenate(true, 0)
    pred_scores = np.concatenate(pred_scores, 0)
    labels_classes = np.concatenate(labels_classes, 0)

    try:
        auroc_macro = metrics.roc_auc_score(labels_classes, pred_scores, average='macro')
        auroc = metrics.roc_auc_score(labels_classes, pred_scores, average=None)
    except ValueError:
        auroc_macro = 0
        auroc = 0

    idx = np.argmax(pred_scores, axis=-1)
    preds_label = np.zeros(pred_scores.shape)
    preds_label[np.arange(preds_label.shape[0]), idx] = 1
    acc = metrics.accuracy_score(labels_classes, preds_label)
    try:
        auprc = metrics.average_precision_score(labels_classes, preds_label, average='macro')
    except ValueError:
        auprc = 0
    return test_loss / pred.shape[0], acc, auroc_macro, auroc, auprc

def evaluate_classifier_wbm(model, test_loader, dec=None, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]

        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                       :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                out = classifier(z0)
                test_loss += nn.BCEWithLogitsLoss()(out, label.float()).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    pred = np.nan_to_num(pred)
    true = np.concatenate(true, 0)
    pred = np.array(pred > 0.5, dtype=float)
    try:
        auroc_macro = metrics.roc_auc_score(true, pred, average='macro')
        auroc = metrics.roc_auc_score(true, pred, average=None)
    except ValueError:
        auroc_macro = 0
        auroc = 0
    acc = metrics.accuracy_score(true, pred)
    try:
        auprc = metrics.average_precision_score(true, pred, average='macro')
    except ValueError:
        auprc = 0
    return test_loss / pred.shape[0], acc, auroc_macro, auroc, auprc

def evaluate_classifier(model, test_loader, dec=None, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]

        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if args.classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x)
                else:
                    out = classifier(z0)
            if args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    pred = np.nan_to_num(pred)
    true = np.concatenate(true, 0)
    acc = np.mean(pred.argmax(1) == true)
    auroc = metrics.roc_auc_score(
        true, pred[:, 1]) if not args.classify_pertp else 0.

    precision, recall, threshold = metrics.precision_recall_curve(true, pred[:, 1])
    auprc = metrics.auc(recall, precision)

    return test_loss / pred.shape[0], acc, auroc, auprc

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

def split(x, y, args):
    kfold = model_selection.StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0)
    splits = [(train_inds, test_inds)
              for train_inds, test_inds in kfold.split(np.zeros(len(y)), y)]
    x_train, y_train = x[splits[args.split][0]], y[splits[args.split][0]]
    test_data_x, test_data_y = x[splits[args.split]
    [1]], y[splits[args.split][1]]
    if not args.old_split:
        train_data_x, val_data_x, train_data_y, val_data_y = \
            model_selection.train_test_split(
                x_train, y_train, stratify=y_train, test_size=0.2, random_state=0)
    else:
        frac = int(0.8 * x_train.shape[0])
        train_data_x, val_data_x = x_train[:frac], x_train[frac:]
        train_data_y, val_data_y = y_train[:frac], y_train[frac:]
    return train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y

def get_data_objects(train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y, input_dim, args):
    train_data_combined = TensorDataset(torch.from_numpy(train_data_x).float(),
                                        torch.from_numpy(train_data_y).long().squeeze())
    val_data_combined = TensorDataset(torch.from_numpy(val_data_x).float(),
                                      torch.from_numpy(val_data_y).long().squeeze())
    test_data_combined = TensorDataset(torch.from_numpy(test_data_x).float(),
                                       torch.from_numpy(test_data_y).long().squeeze())
    train_dataloader = DataLoader(
        train_data_combined, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=args.batch_size, shuffle=True)

    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "val_dataloader": val_dataloader,
                    "input_dim": input_dim}
    return data_objects

def get_mimiciii_data(args):
    x = np.load('../../../Dataset/in_hospital_mortality/input.npy', allow_pickle=True)
    y = np.load('../../../Dataset/in_hospital_mortality/output.npy', allow_pickle=True)
    if args.with_treatment:
        input_dim = 23
        x = x[:, :47]
    else:
        input_dim = 12
        x_1 = x[:, :12]
        x_2 = x[:, 23:35]
        x_3 = x[:, -1]
        x_3 = x_3[:, np.newaxis, :]
        x = np.concatenate((x_1, x_2, x_3), axis=1)
        x = x[:, :25]
    y = y[:]

    x = np.transpose(x, (0, 2, 1))
    # normalize values and time
    process_data(x, input_dim)
    print(x.shape, y.shape)

    train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y = split(x,y,args)

    print(train_data_x.shape, train_data_y.shape, val_data_x.shape, val_data_y.shape,
          test_data_x.shape, test_data_y.shape)
    print(np.sum(test_data_y))

    data_objects = get_data_objects(train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y, input_dim, args)

    return data_objects

def get_decom_data(args):
    x = np.load('../../../Dataset/decom/input.npy', allow_pickle=True)
    y = np.load('../../../Dataset/decom/output.npy', allow_pickle=True)
    if args.with_treatment:
        input_dim = 23
        x = x[:, :47]
    else:
        input_dim = 12
        x_1 = x[:, :12]
        x_2 = x[:, 23:35]
        x_3 = x[:, -1]
        x_3 = x_3[:, np.newaxis, :]
        x = np.concatenate((x_1, x_2, x_3), axis=1)
        x = x[:, :25]
    y = y[:]

    x = np.transpose(x, (0, 2, 1))
    process_data(x, input_dim)
    print(x.shape, y.shape)

    train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y = split(x, y, args)

    print(train_data_x.shape, train_data_y.shape, val_data_x.shape, val_data_y.shape,
          test_data_x.shape, test_data_y.shape)
    print(np.sum(test_data_y))

    data_objects = get_data_objects(train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y,
                                    input_dim, args)
    return data_objects


def get_cip_data(args):
    x = np.load('../../../Dataset/cip/input.npy', allow_pickle=True)
    if args.cip == 'vaso':
        y = np.load('../../../Dataset/cip/vaso_output.npy', allow_pickle=True)
    elif args.cip == 'vent':
        y = np.load('../../../Dataset/cip/vent_output.npy', allow_pickle=True)
    if args.with_treatment:
        input_dim = 23
        x = x[:, :47]
    else:
        input_dim = 12
        x_1 = x[:, :12]
        x_2 = x[:, 23:35]
        x_3 = x[:, -1]
        x_3 = x_3[:, np.newaxis, :]
        x = np.concatenate((x_1, x_2, x_3), axis=1)
        x = x[:, :25]
    y = y[:]


    print(x.shape)
    print(y.shape)

    x = np.transpose(x, (0, 2, 1))
    process_data(x, input_dim)
    print(x.shape, y.shape)

    train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y = split(x, y, args)

    print(train_data_x.shape, train_data_y.shape, val_data_x.shape, val_data_y.shape,
          test_data_x.shape, test_data_y.shape)
    print(np.sum(test_data_y))

    data_objects = get_data_objects(train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y,
                                    input_dim, args)
    return data_objects

def get_los_data(args):
    x = np.load('../../../Dataset/los/input.npy', allow_pickle=True)
    y = np.load('../../../Dataset/los/output.npy', allow_pickle=True)
    if args.with_treatment:
        input_dim = 23
        x = x[:, :47]
    else:
        input_dim = 12
        x_1 = x[:, :12]
        x_2 = x[:, 23:35]
        x_3 = x[:, -1]
        x_3 = x_3[:, np.newaxis, :]
        x = np.concatenate((x_1, x_2, x_3), axis=1)
        x = x[:, :25]
    y = y - 1
    y = y[:]

    x = np.transpose(x, (0, 2, 1))
    process_data(x, input_dim)
    print(x.shape, y.shape)

    train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y = split(x, y, args)

    print(train_data_x.shape, train_data_y.shape, val_data_x.shape, val_data_y.shape,
          test_data_x.shape, test_data_y.shape)
    print(np.sum(test_data_y))

    data_objects = get_data_objects(train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y,
                                    input_dim, args)
    return data_objects

def get_wbm_data(args):
    x = np.load('../../../Dataset/wbm/input.npy', allow_pickle=True)
    y = np.load('../../../Dataset/wbm/output.npy', allow_pickle=True)
    if args.with_treatment:
        input_dim = 23
        x = x[:, :47]
    else:
        input_dim = 12
        x_1 = x[:, :12]
        x_2 = x[:, 23:35]
        x_3 = x[:, -1]
        x_3 = x_3[:, np.newaxis, :]
        x = np.concatenate((x_1, x_2, x_3), axis=1)
        x = x[:, :25]
    y = y[:]

    x = np.transpose(x, (0, 2, 1))
    process_data(x, input_dim)
    print(x.shape, y.shape)

    train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y = split(x, y, args)

    print(train_data_x.shape, train_data_y.shape, val_data_x.shape, val_data_y.shape,
          test_data_x.shape, test_data_y.shape)
    print(np.sum(test_data_y))

    data_objects = get_data_objects(train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y,
                                    input_dim, args)
    return data_objects



