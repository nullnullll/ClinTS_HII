# pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import pandas as pd
from random import SystemRandom
import Baseline.Algorithms.mTAN.models as models
from torch.utils.data import TensorDataset, DataLoader

import Baseline.Algorithms.mTAN.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.1,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--freq', type=float, default=10.)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--dataset', type=str, default='physionet')
parser.add_argument('--alpha', type=int, default=100.)
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--classify-pertp', action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--with-treatment', action='store_true')
parser.add_argument('--early-stop', type=int, default=15)
parser.add_argument('--task', type=str, default='in_hospital_mortality')
parser.add_argument('--cip', type=str, default='vaso')

args = parser.parse_args()

# args.alpha=5
# args.niters=300
# args.lr=0.0001
# args.batch_size=128
# args.rec_hidden=256
# args.gen_hidden=50
# args.latent_dim=128
# args.enc='mtan_rnn'
# args.dec='mtan_rnn'
# args.save=1
# args.classif=True
# args.norm=True
# args.learn_emb=True
# args.k_iwae=1
# args.dataset = 'mimiciii'
# args.seed=0
# args.device = 3
# args.with_treatment = True
# args.task = 'wbm'


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'mimiciii':
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

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]

    rec = models.enc_mtan_rnn(
        dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden,
        embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)

    dec = models.dec_mtan_rnn(
        dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden,
        embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads).to(device)

    if args.task == 'cip':
        classifier = models.create_classifier(args.latent_dim, args.rec_hidden, N=4).to(device)
    elif args.task == 'los':
        classifier = models.create_classifier(args.latent_dim, args.rec_hidden, N=9).to(device)
    elif args.task == 'wbm':
        classifier = models.create_classifier(args.latent_dim, args.rec_hidden, N=12).to(device)
    else:
        classifier = models.create_classifier(args.latent_dim, args.rec_hidden).to(device)

    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    optimizer = optim.Adam(params, lr=args.lr)
    if args.task == 'wbm':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])

    best_val_auroc = float(0)
    total_time = 0.
    times_of_keep_unchange = 0
    stop = False
    for itr in range(1, args.niters + 1):
        if stop == True:
            break
        train_recon_loss, train_ce_loss = 0, 0
        mse = 0
        train_n = 0
        train_acc = 0
        # avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1
        start_time = time.time()
        for train_batch, label in train_loader:
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len = train_batch.shape[0]
            observed_data, observed_mask, observed_tp \
                = train_batch[:, :, :dim], train_batch[:, :, dim:2 * dim], train_batch[:, :, -1]

            out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
            qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
            epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_y = classifier(z0)
            pred_x = dec(
                z0, observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1],
                                 pred_x.shape[2])  # nsample, batch, seqlen, dim
            # compute loss
            logpx, analytic_kl = utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            # label = label.unsqueeze(0).repeat_interleave(args.k_iwae, 0).view(-1)
            if args.task == 'wbm':
                ce_loss = criterion(pred_y, label.float())
            else:
                ce_loss = criterion(pred_y, label)

            loss = recon_loss + args.alpha * ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_ce_loss += ce_loss.item() * batch_len
            train_recon_loss += recon_loss.item() * batch_len
            # train_acc += (pred_y.argmax(1) == label).sum().item()/args.k_iwae
            train_n += batch_len
            mse += utils.mean_squared_error(observed_data, pred_x.mean(0),
                                            observed_mask) * batch_len
        total_time += time.time() - start_time
        if args.task == 'cip' or args.task == 'los':
            val_loss, val_acc, val_auroc, val_auroc_split, val_auprc = utils.evaluate_classifier_cip(
                rec, val_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
            test_loss, test_acc, test_auroc, test_auroc_split, test_auprc = utils.evaluate_classifier_cip(
                rec, test_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        elif args.task == 'wbm':
            val_loss, val_acc, val_auroc, val_auroc_split, val_auprc = utils.evaluate_classifier_wbm(
                rec, val_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
            test_loss, test_acc, test_auroc, test_auroc_split, test_auprc = utils.evaluate_classifier_wbm(
                rec, test_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        else:
            val_loss, val_acc, val_auroc, val_auprc = utils.evaluate_classifier(
                rec, val_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
            test_loss, test_acc, test_auroc, test_auprc = utils.evaluate_classifier(
                rec, test_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)

        print(
            'Iter: {}, recon_loss: {:.4f}, ce_loss: {:.4f}, mse: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, val_auroc: {:.4f}, val_auprc: {:.4f}, test_acc: {:.4f}, test_auroc: {:.4f}, test_auprc: {:.4f}'
            .format(itr, train_recon_loss / train_n, train_ce_loss / train_n, mse / train_n, val_loss, val_acc,
                    val_auroc, val_auprc, test_acc, test_auroc, test_auprc))
        if val_auroc >= best_val_auroc:
            times_of_keep_unchange = 0
            best_val_auroc = max(best_val_auroc, val_auroc)
            best_test_auroc = test_auroc
            best_test_auprc = test_auprc
            if args.task == 'cip' or args.task == 'wbm' or args.task == 'los':
                best_test_auroc_split = test_auroc_split
            rec_state_dict = rec.state_dict()
            dec_state_dict = dec.state_dict()
            classifier_state_dict = classifier.state_dict()
            optimizer_state_dict = optimizer.state_dict()
        else:
            times_of_keep_unchange += 1

        if times_of_keep_unchange > args.early_stop:
            stop = True

