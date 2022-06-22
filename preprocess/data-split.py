import pickle
import numpy as np
import os
import argparse
import random
random.seed(49297)
import pickle
from sklearn import model_selection
import sys

def split(x, y, adm_info, task):
    argsplit = 0
    y_dim = y.shape[-1]
    
    if task == 'wbm':
        y = np.concatenate((y, adm_info),axis=-1)
        x_train, test_data_x,  y_train, test_data_y = model_selection.train_test_split(x, y, test_size=0.20, random_state=42)
    else:
        kfold = model_selection.StratifiedKFold(
            n_splits=5, shuffle=True, random_state=0)

        splits = [(train_inds, test_inds)
                for train_inds, test_inds in kfold.split(np.zeros(len(y)), y)]

        x_train, y_train, train_adm_info = x[splits[argsplit][0]], y[splits[argsplit][0]], adm_info[splits[argsplit][0]]
        test_data_x, test_data_y, test_adm_info = x[splits[argsplit][1]], y[splits[argsplit][1]], adm_info[splits[argsplit][1]]
        

    frac = int(0.8 * x_train.shape[0])
    train_data_x, val_data_x = x_train[:frac], x_train[frac:]
    train_data_y, val_data_y = y_train[:frac], y_train[frac:]
    
    if task == "wbm":
        train_data_y, val_data_y, test_data_y = train_data_y[:,:y_dim], val_data_y[:,:y_dim], test_data_y[:,:y_dim]
        train_adm_info, val_adm_info, test_adm_info = train_data_y[:,y_dim:], val_data_y[:,y_dim:], test_data_y[:,y_dim:]
    else:
        train_adm_info, val_adm_info = train_adm_info[:frac], train_adm_info[frac:]

    return train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y, train_adm_info, val_adm_info, test_adm_info


def get_xy_data(input, output, adm_info_all, task, slice_size=50000):
    train_data_x = []
    val_data_x = []
    test_data_x = []
    train_data_y = []
    val_data_y = []
    test_data_y = []
    
    train_adm_info = []
    val_adm_info = []
    test_adm_info = []
    
    pbar = range(0, len(input), slice_size)
    for start in pbar:
        end = start+slice_size
        if end > len(input):
            end = len(input)
        
        x = input[start:end]
        y = output[start:end]
        adm_info = adm_info_all[start:end]
        
        assert x.shape[0] == y.shape[0]
        
        train_data_x1, val_data_x1, test_data_x1, train_data_y1, val_data_y1, test_data_y1, train_adm_info1, val_adm_info1, test_adm_info1 = split(x, y, adm_info, task)
        
        if len(train_data_x) == 0:
            train_data_x = train_data_x1[:]
            val_data_x = val_data_x1[:]
            test_data_x = test_data_x1[:]
            train_data_y = train_data_y1[:]
            val_data_y = val_data_y1[:]
            test_data_y = test_data_y1[:]
            train_adm_info = train_adm_info1[:]
            val_adm_info = val_adm_info1[:]
            test_adm_info = test_adm_info1[:]
        else:
            train_data_x = np.concatenate((train_data_x, train_data_x1),axis=0)
            val_data_x = np.concatenate((val_data_x, val_data_x1),axis=0)
            test_data_x = np.concatenate((test_data_x, test_data_x1),axis=0)
            train_data_y = np.concatenate((train_data_y, train_data_y1),axis=0)
            val_data_y = np.concatenate((val_data_y, val_data_y1),axis=0)
            test_data_y = np.concatenate((test_data_y, test_data_y1),axis=0)
            train_adm_info = np.concatenate((train_adm_info, train_adm_info1),axis=0)
            val_adm_info = np.concatenate((val_adm_info, val_adm_info1),axis=0)
            test_adm_info = np.concatenate((test_adm_info, test_adm_info1),axis=0)
            

    return (train_data_x, val_data_x, test_data_x, train_data_y, val_data_y, test_data_y, \
        train_adm_info, val_adm_info, test_adm_info)

def save_files(data, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    print("Train:", data[0].shape, data[3].shape, data[6].shape)
    print("Val:",  data[1].shape,  data[4].shape, data[7].shape)
    print("Test:",  data[2].shape,  data[5].shape, data[8].shape)
    print("Saving files...")
    
    pickle.dump( data[6], open(save_path + '/train_sub_adm_icu_idx.p', 'wb'))
    pickle.dump( data[7], open(save_path + '/val_sub_adm_icu_idx.p', 'wb'))
    pickle.dump( data[8], open(save_path + '/test_sub_adm_icu_idx.p', 'wb'))
    
    np.save(save_path + '/train_input.npy',  data[0])
    np.save(save_path + '/val_input.npy',  data[1])
    np.save(save_path + '/test_input.npy',  data[2])
    
    np.save(save_path + '/train_output.npy',  data[3])
    np.save(save_path + '/val_output.npy',  data[4])
    np.save(save_path + '/test_output.npy',  data[5])

    print("Save data splits to ", save_path, " done.")
    

def data_split(task):
    
    data_folder = "./data/" + task
    save_folder = data_folder + "/split"
    
    if not os.path.isdir(data_folder):
        print("Cannot find task data folder: ", data_folder)
        sys.exit(0)
        
    print("#############################################")
    print("Task: ", task)
    
    output_y = None
    input_x = np.load(data_folder + '/input.npy', allow_pickle=True)
    adm_info = pickle.load(open(data_folder + '/sub_adm_icu_idx.p', 'rb'))
    adm_info = np.array(adm_info)
    
    if task != "cip":
        output_y = np.load(data_folder + '/output.npy', allow_pickle=True)
    else:
        vaso_y = np.load(data_folder + '/vaso_output.npy', allow_pickle=True)
        vent_y = np.load(data_folder + '/vent_output.npy', allow_pickle=True)
    
    if task == "decom" or task == "in_hospital_mortality":
        data = split(input_x, output_y, adm_info, task)
        save_files(data, save_folder)
    elif task == "cip":
        data = get_xy_data(input_x, vent_y, adm_info, "vent")
        save_files(data, save_folder+"_vent")
        data = get_xy_data(input_x, vaso_y, adm_info, "vaso")
        save_files(data, save_folder+"_vaso")
    else:
        data = get_xy_data(input_x, output_y, adm_info, task)
        save_files(data, save_folder)

if __name__ == '__main__':

    ####### Obtaining data from database #######
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    args = parser.parse_args()
    
    print("Begin data splitting...")
    
    if args.task is None:
        for task in ["in_hospital_mortality", "decom", "cip", "wbm", "los"]:
            data_split(task)
    else:
        data_split(args.task)
    
    print("Done")
