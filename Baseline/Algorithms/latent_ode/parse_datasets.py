###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
import Baseline.Algorithms.latent_ode.utils as utils
from torch.utils.data import DataLoader
from sklearn import model_selection
import random

#####################################################################################################
def parse_datasets(args, device):
	

	def basic_collate_fn(batch, time_steps, args = args, device = device, data_type = "train"):
		batch = torch.stack(batch)
		data_dict = {
			"data": batch, 
			"time_steps": time_steps}

		data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
		return data_dict


	dataset_name = args.dataset

	n_total_tp = args.timepoints + args.extrap
	max_t_extrap = args.max_t / args.timepoints * n_total_tp


	input_dim = 23
	x = np.load('../Dataset/' + args.task + '/input.npy', allow_pickle=True)
	y = np.load('../Dataset/' + args.task + '/output.npy', allow_pickle=True)

	x = np.transpose(x, (0, 2, 1))

	# normalize values and time
	observed_vals, observed_mask, observed_tp = x[:, :,
												:input_dim], x[:, :, input_dim:2 * input_dim], x[:, :, -1]
	if np.max(observed_tp) != 0.:
		observed_tp = observed_tp / np.max(observed_tp)
	observed_vals[observed_mask == 0] = 0
	kfold = model_selection.StratifiedKFold(
		n_splits=5, shuffle=True, random_state=0)
	splits = [(train_inds, test_inds)
			  for train_inds, test_inds in kfold.split(np.zeros(len(y)), y)]
	x_train, y_train = x[splits[0][0]], y[splits[0][0]]
	test_data_x, test_data_y = x[splits[0][1]], y[splits[0][1]]

	train_data_x = x_train
	train_data_y = y_train

	train_data_x = torch.from_numpy(train_data_x).float()
	train_data_tt = train_data_x[:, :, -1]
	train_data_vals = train_data_x[:, :, :25]
	train_data_mask = train_data_x[:, :, 25:50]
	train_data_y = torch.from_numpy(train_data_y).long().squeeze()
	train_data_label = train_data_y

	test_data_x = torch.from_numpy(test_data_x).float()
	test_data_tt = test_data_x[:, :, -1]
	test_data_vals = test_data_x[:, :, :25]
	test_data_mask = test_data_x[:, :, 25:50]
	test_data_y = torch.from_numpy(test_data_y).long().squeeze()
	test_data_label = test_data_y

	train_data = []
	test_data = []

	for i in range(len(train_data_x)):
		tt = train_data_tt[i].to(device)
		vals = train_data_vals[i].to(device)
		mask = train_data_mask[i].to(device)
		labels = train_data_label[i].to(device)
		t = ("", tt,vals,mask,labels)
		train_data.append(t)

	for i in range(len(test_data_x)):
		tt = test_data_tt[i].to(device)
		vals = test_data_vals[i].to(device)
		mask = test_data_mask[i].to(device)
		labels = test_data_label[i].to(device)
		t = ("", tt,vals,mask,labels)
		test_data.append(t)

	total_dataset=train_data+test_data

	record_id, tt, vals, mask, labels = train_data[0]

	n_samples = len(total_dataset)
	input_dim = vals.size(-1)

	batch_size = args.batch_size
	data_min, data_max = get_data_min_max(total_dataset)


	train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False,
		collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
			data_min = data_min, data_max = data_max))
	test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False,
		collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
			data_min = data_min, data_max = data_max))


	data_objects = {"dataset_obj": total_dataset,
				"train_dataloader": utils.inf_generator(train_dataloader),
				"test_dataloader": utils.inf_generator(test_dataloader),
				"input_dim": input_dim,
				"n_train_batches": len(train_dataloader),
				"n_test_batches": len(test_dataloader),
				"classif_per_tp": False, #optional
				"n_labels": 1} #optional
	return data_objects




