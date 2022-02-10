import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn import model_selection
from sklearn import metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    x = np.load('/home/covpreduser/Blob/v-chong/pycharm_projects/mTAN/final_input_event_50000.npy', allow_pickle=True)
    y = np.load('/home/covpreduser/Blob/v-chong/pycharm_projects/mTAN/final_output_event_50000.npy', allow_pickle=True)
    if args.with_treatment:
        input_dim = 25
        x = x[:1000, :51]
    else:
        input_dim = 12
        x_1 = x[:, :12]
        x_2 = x[:, 25:37]
        x_3 = x[:, -1]
        x_3 = x_3[:, np.newaxis, :]
        x = np.concatenate((x_1, x_2, x_3), axis=1)
        x = x[:1000, :25]
    y = y[:1000]

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
    x = np.load('../Dataset/decom_13w/input.npy', allow_pickle=True)
    y = np.load('../Dataset/decom_13w/output.npy', allow_pickle=True)
    if args.with_treatment:
        input_dim = 25
        x = x[:, :51]
    else:
        input_dim = 12
        x_1 = x[:, :12]
        x_2 = x[:, 25:37]
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
    x = np.load('../Dataset/cip/input.npy', allow_pickle=True)
    if args.cip == 'vaso':
        y = np.load('../Dataset/cip/vaso_output.npy', allow_pickle=True)
    elif args.cip == 'vent':
        y = np.load('../Dataset/cip/vent_output.npy', allow_pickle=True)
    if args.with_treatment:
        input_dim = 25
        x = x[:, :51]
    else:
        input_dim = 12
        x_1 = x[:, :12]
        x_2 = x[:, 25:37]
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
    x = np.load('../Dataset/los/input.npy', allow_pickle=True)
    y = np.load('../Dataset/los/output.npy', allow_pickle=True)
    if args.with_treatment:
        input_dim = 25
        x = x[:, :51]
    else:
        input_dim = 12
        x_1 = x[:, :12]
        x_2 = x[:, 25:37]
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
    x = np.load('../Dataset/wbm/input.npy', allow_pickle=True)
    y = np.load('../Dataset/wbm/output.npy', allow_pickle=True)
    if args.with_treatment:
        input_dim = 25
        x = x[:, :51]
    else:
        input_dim = 12
        x_1 = x[:, :12]
        x_2 = x[:, 25:37]
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

def evaluation_matrics(pred, true ):

    matrics = []
    mse = []
    mape = []
    nd = []
    nrmse = 0
    mse.append(sklearn.metrics.mean_squared_error(true, pred))
    matrics.append(mse)

    mask = true != 0
    mape.append(np.fabs((true[mask] - pred[mask]) / true[mask]).mean())
    matrics.append(mape)

    nd.append(mean_absolute_error(true, pred) / true.mean())
    matrics.append(nd)

    nrmse = (np.sqrt(mse) / true.mean())
    matrics.append(nrmse)

    return matrics

def getFileNameandPath(args):
    name = args.enc + '_'
    save_path = '../checkpoint/'
    if args.with_treatment:
        name += 'with_treatment_'
        if args.causal_masking:
            name += 'causal_mask_'
        save_path += args.enc + '_with_treatment/'
    else:
        save_path += args.enc+ '/'
    name += args.task  + '_' + str(args.seed)
    return name,save_path

def mean_squared_error(orig, pred):
    error = (orig - pred) ** 2
    return error.sum() / pred.shape[0]

def calculate_time_loss(prediction, event_time, mask, device):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(device)
    true = event_time[:, 1:]
    prediction = prediction[:, 1:]
    diff = prediction[mask[:,1:]] - true[mask[:,1:]]
    return -torch.sum(-.5 * (torch.log(const)  + diff ** 2.)) /  diff.shape[0]

def evaluate_classifier(rec,dec, test_loader, args, classifier=None, dim=12, device='cuda', num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp = \
            test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]
        if args.with_treatment:
            if args.causal_masking:
                causal_mask = get_causal_mask(observed_mask, observed_tp, dim, device)
                causal_mask = causal_mask[:, :, :, 12:25]
            else:
                causal_mask = None
        else:
            causal_mask = None
        with torch.no_grad():

            ClsInput = rec(torch.cat((observed_data, observed_mask), 2), observed_tp, causal_mask = causal_mask)
            out = classifier(ClsInput, args)

            test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    pred = np.nan_to_num(pred)
    true = np.concatenate(true, 0)
    acc = np.mean(pred.argmax(1) == true)
    auroc = metrics.roc_auc_score(
        true, pred[:, 1])

    precision, recall, threshold = metrics.precision_recall_curve(true, pred[:, 1])
    auprc = metrics.auc(recall, precision)

    return test_loss/pred.shape[0], acc, auroc, auprc

def evaluate_classifier_cip(rec,dec, test_loader, args, classifier=None, dim=12, device='cuda', num_sample=1):
    pred = []
    true = []
    pred_scores = []
    labels_classes = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp = \
            test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]
        if args.with_treatment:
            if args.causal_masking:
                causal_mask = get_causal_mask(observed_mask, observed_tp, dim, device)
                causal_mask = causal_mask[:, :, :, 12:25]
            else:
                causal_mask = None
        else:
            causal_mask = None
        with torch.no_grad():

            ClsInput = rec(torch.cat((observed_data, observed_mask), 2), observed_tp, causal_mask = causal_mask)
            out = classifier(ClsInput, args)
            scores = torch.softmax(out, dim=1)
            test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
            if args.task == 'cip':
                label_classes = label_binarize(label.cpu().numpy(), classes=range(4))
            else:
                label_classes = label_binarize(label.cpu().numpy(), classes=range(9))
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
    auprc = metrics.average_precision_score(labels_classes, preds_label, average='macro')

    return test_loss/pred.shape[0], acc, auroc_macro, auroc, auprc

def evaluate_classifier_wbm(rec,dec, test_loader, args, classifier=None, dim=12, device='cuda', num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp = \
            test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]
        if args.with_treatment:
            if args.causal_masking:
                causal_mask = get_causal_mask(observed_mask, observed_tp, dim, device)
                causal_mask = causal_mask[:, :, :, 12:25]
            else:
                causal_mask = None
        else:
            causal_mask = None
        with torch.no_grad():

            ClsInput = rec(torch.cat((observed_data, observed_mask), 2), observed_tp, causal_mask=causal_mask)
            out = classifier(ClsInput, args)
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

def get_causal_mask(observed_mask, observed_tp, dim, device):
    ref_time_stamps = torch.linspace(0, 1., 128).to(device)
    time_mask = torch.transpose(observed_mask, 1, 2).contiguous().to(device)
    time_stamps = observed_tp.view(observed_tp.size(0), observed_tp.size(1))
    time_stamps = time_stamps.unsqueeze(-2).repeat_interleave(dim, dim=-2)
    time_stamps = time_stamps.masked_fill(time_mask == 0, 10)
    ref_time_stamps = ref_time_stamps[None, None, :, None].expand(time_mask.size(0), time_mask.size(1), 128, 200)
    time_stamps = time_stamps.unsqueeze(-2).expand(time_mask.size(0), time_mask.size(1), 128, 200)
    causal_mask = time_stamps < ref_time_stamps
    causal_mask = causal_mask.permute(0, 2, 3, 1)
    causal_mask = causal_mask.to(device)

    return  causal_mask


