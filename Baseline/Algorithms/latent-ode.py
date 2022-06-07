###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys
import matplotlib
from torch.distributions.normal import Normal
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import Baseline.Algorithms.latent_ode.utils as utils


from Baseline.Algorithms.latent_ode.rnn_baselines import *
from Baseline.Algorithms.latent_ode.ode_rnn import *
from Baseline.Algorithms.latent_ode.create_latent_ode_model import create_LatentODE_model
from Baseline.Algorithms.latent_ode.parse_datasets import parse_datasets


from Baseline.Algorithms.latent_ode.utils import compute_loss_all_batches

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n', type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None,
                    help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='periodic',
                    help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
                                                                        "If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None,
                    help="Cut out the section of the timeline of the specified length (in number of points)."
                         "Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
                                                                    "Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--latent_ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn',
                    help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true',
                    help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true',
                    help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true',
                    help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true',
                    help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100,
                    help="Number of units per layer in each of GRU update networks")

parser.add_argument('--poisson', action='store_true',
                    help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', action='store_true',
                    help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true',
                    help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t', type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")
parser.add_argument('--task', type=str, default='in_hospital_mortality')

args = parser.parse_args()

args.niters = 100
args.l = 20
args.n = 8000
args.latent_ode = True
args.rec_dims = 40
args.rec_layers = 3
args.gen_layers = 3
args.units = 50
args.gru_units = 50
args.quantization = 0.016
args.classif = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

    start = time.time()
    print("Sampling dataset of {} training examples".format(args.n))

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    utils.makedirs("results/")

    ##################################################################
    data_obj = parse_datasets(args, device)
    input_dim = data_obj["input_dim"]

    classif_per_tp = False
    if ("classif_per_tp" in data_obj):
        # do classification per time point rather than on a time series as a whole
        classif_per_tp = data_obj["classif_per_tp"]

    if args.classif and (args.dataset == "hopper" or args.dataset == "periodic"):
        raise Exception("Classification task is not available for MuJoCo and 1d datasets")

    n_labels = 1
    if args.classif:
        if ("n_labels" in data_obj):
            n_labels = data_obj["n_labels"]
        else:
            raise Exception("Please provide number of labels for classification task")

    ##################################################################
    # Create the model
    obsrv_std = 0.01
    if args.dataset == "hopper":
        obsrv_std = 1e-3

    obsrv_std = torch.Tensor([obsrv_std]).to(device)

    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device,
                                   classif_per_tp=classif_per_tp,
                                   n_labels=n_labels)


    # Load checkpoint and evaluate the model
    if args.load is not None:
        utils.get_ckpt_model(ckpt_path, model, device)
        exit()

    ##################################################################
    # Training

    log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    num_batches = data_obj["n_train_batches"]
    print("num_batches: " + str(num_batches))
    print("Epoch num: " + str(num_batches * (args.niters + 1)))
    for itr in range(1, num_batches * (args.niters + 1)):
        optimizer.zero_grad()
        utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr / 10)

        wait_until_kl_inc = 10
        if itr // num_batches < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1 - 0.99 ** (itr // num_batches - wait_until_kl_inc))

        batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
        train_res = model.compute_all_losses(batch_dict, n_traj_samples=3, kl_coef=kl_coef)
        train_res["loss"].backward()
        optimizer.step()

        n_iters_to_viz = 1
        if itr % (n_iters_to_viz * num_batches) == 0:

            with torch.no_grad():

                test_res = compute_loss_all_batches(model,
                                                    data_obj["test_dataloader"], args,
                                                    n_batches=data_obj["n_test_batches"],
                                                    experimentID=experimentID,
                                                    device=device,
                                                    n_traj_samples=3, kl_coef=kl_coef)

                message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                    itr // num_batches,
                    test_res["loss"].detach(), test_res["likelihood"].detach(),
                    test_res["kl_first_p"], test_res["std_first_p"])

                logger.info("Experiment " + str(experimentID))
                logger.info(message)
                logger.info("KL coef: {}".format(kl_coef))
                logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
                logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))

                # print('Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'
                # 	  .format(itr//num_batches, test_res["loss"].detach(), test_res["likelihood"].detach(), test_res["kl_first_p"], test_res["std_first_p"]))

                if "auc" in test_res:
                    logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))
                # print("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))
                if "mse" in test_res:
                    logger.info("Test MSE: {:.4f}".format(test_res["mse"]))
                # print("Test MSE: {:.4f}".format(test_res["mse"]))
                if "accuracy" in train_res:
                    logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))
                # print("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))
                if "accuracy" in test_res:
                    logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))
                # print("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))
                if "pois_likelihood" in test_res:
                    logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))
                # print("Poisson likelihood: {}".format(test_res["pois_likelihood"]))
                if "ce_loss" in test_res:
                    logger.info("CE loss: {}".format(test_res["ce_loss"]))
                # print("CE loss: {}".format(test_res["ce_loss"]))
            torch.save({
                'args': args,
                'state_dict': model.state_dict(),
            }, ckpt_path)


    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)

