'''
Part of this script is modified upon MIMIC-Extract (https://github.com/MLforHealth/MIMIC_Extract) and IP-Nets (https://github.com/mlds-lab/interp-net).
'''

import pickle
import psycopg2 as py
import copy
import numpy as np
import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm, trange
import pickle 

import proc_util
from proc_util.task_build import *
from proc_util.extract_cip_label import *

min_time_period = 48

def load_adm2sbj(subject2hadm):
    # subj2adm dict
    adm2subj_dict = {}
    for subject_id, hadm_id in subject2hadm:
        if hadm_id not in adm2subj_dict:
            adm2subj_dict.update({hadm_id:subject_id})
        else:
            print("admid: ", str(hadm_id), "subject:", adm2subj_dict[hadm_id], subject_id)
            continue
    return adm2subj_dict

def load_adm2icu(icustays):
    icu_dict = {}
    los_dict = {}
    for hadm_id, icustay_id, intime, outtime, los in icustays:
        if hadm_id not in icu_dict:
            icu_dict.update({hadm_id:{icustay_id:[intime, outtime]}})
            los_dict.update({str(hadm_id)+'_'+str(icustay_id) : los})
        elif icustay_id not in icu_dict[hadm_id]:
            icu_dict[hadm_id].update({icustay_id:[intime, outtime]})
            los_dict.update({str(hadm_id)+'_'+str(icustay_id): los})
        else:
            continue
    return icu_dict, los_dict

def load_adm2deathtime(deathtimes):
    adm2deathtime_dict = {}
    for hadm_id, deathtime in deathtimes:
        if hadm_id not in adm2deathtime_dict:
            adm2deathtime_dict.update({hadm_id:deathtime})
    return deathtimes


def load_adm_data(data_path):
    print('Loading files from ', data_path, ' ...')
    adm_info = pickle.load(
            open(data_path, 'rb'))
    adm_id_needed = [record[0] for record in adm_info if record[2] >= min_time_period]

    print('Loading ', str(len(adm_info)), ' adm Done!')
    return adm_info, adm_id_needed


def load_data(data_path, adm_info, adm_id_needed=None):
   
    print('Loading files from ', data_path, ' ...')
    vitals = pickle.load(open(data_path, 'rb'))
    print('Loading Done!')
    
    adm_id = [record[0] for record in adm_info]
    
    if adm_id_needed is None:
        adm_id_needed = [record[0] for record in adm_info if record[2] >= min_time_period]

    vitals_dict = {}
    for i in range(len(vitals)):
        vitals_dict[adm_id[i]] = vitals[i]

    vitals = []
    label = []
    adm_id__vital = []
    for x in adm_id_needed:
        if x in vitals_dict:
            vitals.append(vitals_dict[x])
            adm_id__vital.append(x)
            for rec in adm_info:
                if x == rec[0]:
                    label.append(rec[3])
                    break

    assert len(vitals) == len(label)

    return adm_id__vital, vitals, label

def select_features(data):
    for i in range(len(data)):
        for elem in data[i][7]:
            if elem[1] != None:
                # Fahrenheit->Celcius conversion
                tup = (elem[0], elem[1] * 1.8 + 32)
                data[i][6].append(tup)

        for elem in data[i][10]:
            data[i][9].append(elem)
        for elem in data[i][11]:
            data[i][9].append(elem)

        # removing duplicates and EtCO2
        del data[i][5]
        del data[i][6]
        del data[i][8]
        del data[i][8]

        for j in range(len(data[i][-1])):
            data[i][-1][j] = (data[i][-1][j][0], 1)

        for j in range(len(data[i][-2])):
            data[i][-2][j] = (data[i][-2][j][0], 1)

        del data[i][14]
        del data[i][18]
    
    print("feature select done")


def trim_los(data, shortest_length=48):
    """Used to build time set
    """
    num_features = 23  # final features (excluding EtCO2)
    max_length = 2881  # maximum length of time stamp(48 * 60)
    a = np.zeros((len(data), num_features, max_length))
    timestamps, rawdata, rawtimestamps = [], [], []

    for i in range(len(data)):
        l = []
        # taking union of all time stamps,
        # we don't actually need this for our model
        for j in range(num_features):
            for k in range(len(data[i][j])):
                l.append(data[i][j][k][0])

        # keeping only unique elements
        TS = []
        for j in l:
            if j not in TS:
                TS.append(j)
        TS.sort()

        temp = []
        for t in TS:
            if (t - TS[0]).total_seconds() / 3600 <= shortest_length:
                temp.append(t)

        # extracting first 48hr data
        T = copy.deepcopy(TS)
        TS = []
        for t in T:
            if (t - T[0]).total_seconds() / 3600 <= shortest_length:
                TS.append(t)
        T = []
        timestamps.append(TS)
        rawdata.append(data[i])
        rawtimestamps.append(temp)

        for j in range(num_features):
            c = 0
            for k in range(len(TS)):
                if c < len(data[i][j]) and TS[k] == data[i][j][c][0]:
                    if data[i][j][c][1] is None:
                        a[i, j, k] = -100  # missing data
                    elif (data[i][j][c][1] == 'Normal <3 secs' or
                          data[i][j][c][1] == 'Normal <3 Seconds' or
                          data[i][j][c][1] == 'Brisk'):
                        a[i, j, k] = 1
                    elif (data[i][j][c][1] == 'Abnormal >3 secs' or
                          data[i][j][c][1] == 'Abnormal >3 Seconds' or
                          data[i][j][c][1] == 'Delayed'):
                        a[i, j, k] = 2
                    elif (data[i][j][c][1] == 'Other/Remarks' or
                          data[i][j][c][1] == 'Comment'):
                        a[i, j, k] = -100  # missing data
                    else:
                        if (j == 21 or j == 22):
                            a[i, j, k] = 1
                        else:
                            a[i, j, k] = data[i][j][c][1]

                    c += 1
                else:
                    a[i, j, k] = -100  # missing data

    print("feature extraction success")
    print("value processing success ")
    return a, timestamps,rawdata, rawtimestamps

def fix_input_format(x, T):
    """Return the input in the proper format
    x: observed values
    M: masking, 0 indicates missing values
    delta: time points of observation
    """
    timestamp = 200
    num_features = 23

    # trim time stamps higher than 200
    for i in range(len(T)):
        if len(T[i]) > timestamp:
            T[i] = T[i][:timestamp]

    x = x[:, :, :timestamp]
    M = np.zeros_like(x)
    delta = np.zeros_like(x)
    print(x.shape, len(T))

    for t in T:
        for i in range(1, len(t)):
            t[i] = (t[i] - t[0]).total_seconds()/3600.0
        if len(t) != 0:
            t[0] = 0

    # count outliers and negative values as missing values
    # M = 0 indicates missing value
    # M = 1 indicates observed value
    # now since we have mask variable, we don't need -100

    M[x > 500] = 0
    x[x > 500] = 0.0
    M[x < 0] = 0
    x[x < 0] = 0.0
    M[x > 0] = 1

    for i in range(num_features):
        for j in range(x.shape[0]):
            for k in range(len(T[j])):
                delta[j, i, k] = T[j][k]


    return x, M, delta

def save_xy(in_x,in_m,in_T, label, save_path):
    
    x = np.concatenate((in_x,in_m,in_T) , axis=1)  # input format
    y = np.array(label)
    np.save(save_path + 'input.npy', x)
    np.save(save_path + 'output.npy', y)
    print(x.shape)
    print(y.shape)

    print(save_path, " saved success")

def preproc_xy(adm_icu_id, data_x, data_y, shortest_length, dataset_name):

    out_value, out_timestamps, _, _ = trim_los(data_x, shortest_length)

    x, m, T = fix_input_format(out_value, out_timestamps)
    print("timestamps format processing success")

    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)
    
    pickle.dump(adm_icu_id, open(dataset_name + 'sub_adm_icu_idx.p', 'wb'))
    save_xy(x, m, T, data_y, dataset_name)
    
    
def save_interv_xy(in_x,in_m,in_T, vent_label, vaso_label, save_path):
    
    x = np.concatenate((in_x,in_m,in_T) , axis=1)  # input format
    vent_label = np.array(vent_label)
    vaso_label = np.array(vaso_label)
    np.save(save_path + 'input.npy', x)
    np.save(save_path + 'vent_output.npy', vent_label)
    np.save(save_path + 'vaso_output.npy', vaso_label)
    print(x.shape)
    print(vent_label.shape)
    print(vaso_label.shape)

    print(save_path, " saved success")

def preproc_interv_xy(adm_icu_id, data_x, vent_label, vaso_label, shortest_length, dataset_name):

    out_value, out_timestamps, _, _ = trim_los(data_x, shortest_length)

    x, m, T = fix_input_format(out_value, out_timestamps)
    print("timestamps format processing success")
    
    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)
        
    pickle.dump(adm_icu_id, open(dataset_name + 'sub_adm_icu_idx.p', 'wb'))
    save_interv_xy(x, m, T, vent_label, vaso_label, dataset_name)
    
    
def data_shuffling(vitals, adm_id_needed, label):
    tmp_data = list(zip(vitals, adm_id_needed, label))
    random.shuffle(tmp_data)

    rd_vitals = [i[0] for i in tmp_data]
    rd_adm_id_needed = [i[1] for i in tmp_data]
    rd_label = [i[2] for i in tmp_data]

    vitals, adm_id_needed, label = rd_vitals, rd_adm_id_needed, rd_label

    print("shuffling done")
    return vitals, adm_id_needed, label 

if __name__ == '__main__':
    data_tmp_folder = "./data/tmp/"
    adm_id_data_path = data_tmp_folder + "adm_type_los_mortality.p"
    bio_path = data_tmp_folder + "patient_records.p"
    interv_outPath = data_tmp_folder + "cip_hourly_data.h5"
    resource_path = "./preprocess/proc_util/resource/"
    

    ####### Obtaining data from database #######
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', type=str, default='mimic')
    parser.add_argument('--user', type=str, default='postgres')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--password', type=str, default='postgres')
    parser.add_argument('--search_path', type=str, default='mimiciii')
    args = parser.parse_args()

    print("connecting database...")
    connect_str = "dbname=" + args.dbname + " user=" + args.user + " host=" + args.host + " password=" + args.password + " options=--search_path=" + args.search_path
    print(connect_str)
    conn = py.connect(connect_str)

    cur = conn.cursor()
    # load icu stay
    print("loading icustay data...")
    cur.execute("select hadm_id, icustay_id, intime, outtime, los from icustays")
    icustays = cur.fetchall()

    print("loading subject data...")
    # load subject2hadm
    cur.execute("select  subject_id, hadm_id from admissions")
    subject2hadm = cur.fetchall()

    # subj2adm dict
    adm2subj_dict = load_adm2sbj(subject2hadm)
    print("load ", str(len(subject2hadm)), " admission data")
    
    # adm2icu dict
    icu_dict, los_dict = load_adm2icu(icustays)
    print("load ", str(len(icu_dict)), " icu data")
    
    # death time
    cur.execute("""select hadm_id, deathtime from admissions """)
    deathtimes = cur.fetchall()
    adm2deathtime_dict = load_adm2deathtime(deathtimes)
    print("load ", str(len(adm2deathtime_dict)), " deathtime data")
    
    ####### Load data #######
    adm_info, adm_id_needed = load_adm_data(adm_id_data_path)
    
    # load biomarkers and events
    adm_id_needed, vitals, label = load_data(bio_path, adm_info, adm_id_needed=adm_id_needed)
    
    # select features
    select_features(vitals)
    print("after selection, ", str(len(vitals[0])), " feature left.")
    
    
    # data shuffling (optional)
    # vitals, adm_id_needed, label = data_shuffling(vitals, adm_id_needed, label)
    
    ####### Task Building #######
    
    #==== In-hospital mortality ====
    print("building in-hospital mortality task...")
    
    mor_adm_icu_id, mor_data, mor_label = adm_id_needed, vitals, label
    
    preproc_xy(mor_adm_icu_id, mor_data, mor_label, 48, './data/in_hospital_mortality/')
    
    #==== Decompensation ====
    print("building Decompensation task...")
    
    decom_adm_icu_id, decom_data, decom_label = create_decompensation(vitals, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, adm_id_needed, label, sample_rate=6.0, shortest_length=12.0, future_time_interval=24.0, start=0, end=len(adm_id_needed), need_sample=-1)
    
    preproc_xy(decom_adm_icu_id, decom_data, decom_label, 12, './data/decom/')
    
    
    #==== Length of Stay ====
    print("building Length of Stay task...")
    
    los_adm_icu_id, los_data, los_label = create_los(vitals, icu_dict, los_dict, adm2subj_dict, adm_id_needed, label, sample_rate=12, shortest_length=24, eps=1e-6, start=0, end=len(adm_id_needed), need_sample=-1)
    
    preproc_xy(los_adm_icu_id, los_data, los_label, 48, './data/los/')
    
    #==== Next Timepoint Will be Measured ====
    print("building Next Timepoint Will be Measured task...")
    
    wbm_adm_icu_id, wbm_data, wbm_label = create_wbm(vitals, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, adm_id_needed, sample_rate=12.0, observ_win=48.0, eps=1e-6, future_time_interval=1.0, start=0, end=len(adm_id_needed), need_sample=-1)
    
    preproc_xy(wbm_adm_icu_id, wbm_data, wbm_label, 48, './data/wbm/')
    
    #==== Clinical Intervention Prediction ====
    print("building Clinical Intervention Prediction task...")
    
    if not os.path.isfile(interv_outPath):
        extract_cip_data(resource_path, interv_outPath, args.dbname, args.search_path, args.host, args.user, args.password)
    
    print("Loading files from ", interv_outPath)
    Y = pd.read_hdf(interv_outPath,'interventions')
    Y = Y[['vent', 'vaso']]
    
    cip_adm_icu_id, cip_data, vent_labels, vaso_labels = create_interv_pred(vitals, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict, adm_id_needed, Y, sample_rate=6, observ_win=6, eps=1e-6, future_time_interval=4, gap_win=6, start=0, end=len(vitals), need_sample=-1)
    
    preproc_interv_xy(cip_adm_icu_id, cip_data, vent_labels, vaso_labels, 48, './data/cip/')
    
    
    print("build all the task done.")