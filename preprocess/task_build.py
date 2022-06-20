import sys
import numpy as np
from tqdm import tqdm, trange

def create_decompensation(data, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, adm_ids, label, sample_rate=1.0, shortest_length=4.0, eps=1e-6, future_time_interval=24.0, start=0, end=10000, need_sample=-1):
    adm_icu_id = []
    decom_data = []
    decom_label = []

    for i in trange(start, end):
        if need_sample != -1 and len(decom_data) > need_sample:
            break

        mortality = label[i]

        # filter
        if mortality < 1:
            continue

        adm_data = data[i] # (23 * T_feature)
        if adm_ids[i] not in icu_dict:
            continue
        for icustay_id in icu_dict[adm_ids[i]]:
            icu_data = []
            # empty label
            if label[i] is None:
                continue

            if pd.isnull(los_dict[str(adm_ids[i])+'_'+str(icustay_id)]):
                #print("(length of stay is missing)", adm_ids[i], icustay_id)
                continue

            los = 24.0 * los_dict[str(adm_ids[i])+'_'+str(icustay_id)]  # in hours

            deathtime = adm2deathtime_dict[adm_ids[i]]
            intime = icu_dict[adm_ids[i]][icustay_id][0]
            outtime = icu_dict[adm_ids[i]][icustay_id][1]

            if deathtime is None:
                lived_time = 1e18
            else:
                lived_time = (deathtime - intime).total_seconds() / 3600.0

            for k in range(len(adm_data)):
                kth_feature = []
                for t, value in adm_data[k]:
                    if intime < t < outtime:
                        kth_feature.append((t, value))

                icu_data.append(kth_feature)


            sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate) 
            sample_times = list(filter(lambda x: x > shortest_length, sample_times))

            for t in sample_times:
                sample_data = []

                # get label
                if mortality == 0:
                    cur_mortality = 0
                else:
                    cur_mortality = int(lived_time - t < future_time_interval)

                # get data
                for feature in range(len(icu_data)):
                    f = []
                    for ft,val in icu_data[feature]:
                        if t <= (ft - intime).total_seconds() / 3600.0 < t + shortest_length:
                            f.append((ft, val))

                    sample_data.append(f)

                adm_icu_id.append((adm2subj_dict[adm_ids[i]], adm_ids[i], icustay_id))
                decom_data.append(sample_data)
                decom_label.append(cur_mortality) 

    print("Number of created samples:", len(decom_data), len(adm_icu_id), len(decom_label))
    print("Number of features:", len(decom_data[0]))
    return adm_icu_id, decom_data, decom_label


def number2cls(number):
        
    if number <= 7.0:
        if number <= 0:
            number = 1
        return math.ceil(number)
    elif 7 < number < 14:
        return 8
    else:
        return 9

def create_los(data, icu_dict, los_dict, adm2subj_dict, adm_ids, label, sample_rate=1.0, shortest_length=4.0, eps=1e-6, start=0, end=10000, need_sample=-1):
    adm_icu_id = []
    decom_data = []
    decom_label = []

    for i in trange(start, end):
        if need_sample != -1 and len(decom_data) > need_sample:
            break

        adm_data = data[i] # (23 * T_feature)
        if adm_ids[i] not in icu_dict:
            continue
        for icustay_id in icu_dict[adm_ids[i]]:
            icu_data = []
            # empty label
            if label[i] is None:
                continue

            if pd.isnull(los_dict[str(adm_ids[i])+'_'+str(icustay_id)]):
                #print("(length of stay is missing)", adm_ids[i], icustay_id)
                continue

            los = 24.0 * los_dict[str(adm_ids[i])+'_'+str(icustay_id)]  # in hours


            intime = icu_dict[adm_ids[i]][icustay_id][0]
            outtime = icu_dict[adm_ids[i]][icustay_id][1]

            for k in range(len(adm_data)):
                kth_feature = []
                for t, value in adm_data[k]:
                    if intime < t < outtime:
                        kth_feature.append((t, value))

                icu_data.append(kth_feature)


            sample_times = np.arange(0.0, los + eps, sample_rate) 

            sample_times = list(filter(lambda x: x > shortest_length, sample_times))

            for t in sample_times:
                sample_data = []

                # get data
                for feature in range(len(icu_data)):
                    f = []
                    for ft,val in icu_data[feature]:
                        if t <= ((ft - intime).total_seconds() / 3600.0) < t + shortest_length:
                            f.append((ft, val))

                    sample_data.append(f)

                cur_label = number2cls((los - t)/24)

                adm_icu_id.append((adm2subj_dict[adm_ids[i]], adm_ids[i], icustay_id))
                decom_data.append(sample_data)
                decom_label.append(cur_label) 

    print("Number of created samples:", len(decom_data), len(adm_icu_id), len(decom_label))
    print("Number of features:", len(decom_data[0]))
    return adm_icu_id, decom_data, decom_label


def create_wbm(data, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, adm_ids, sample_rate=1.0, observ_win=4.0, eps=1e-6, future_time_interval=1.0, start=0, end=10000, need_sample=-1):
    adm_icu_id = []
    wbm_data = []
    wbm_label = []

    # go through icu stays
    for i in trange(start, end):
        if need_sample != -1 and len(wbm_data) > need_sample:
            break

        adm_data = data[i] # (23 * T_feature)
        if adm_ids[i] not in icu_dict:
            continue
        for icustay_id in icu_dict[adm_ids[i]]:
            icu_data = []


            if pd.isnull(los_dict[str(adm_ids[i])+'_'+str(icustay_id)]):
                continue

            los = 24.0 * los_dict[str(adm_ids[i])+'_'+str(icustay_id)]  # in hours

            deathtime = adm2deathtime_dict[adm_ids[i]]
            intime = icu_dict[adm_ids[i]][icustay_id][0]
            outtime = icu_dict[adm_ids[i]][icustay_id][1]

            if deathtime is None:
                lived_time = (outtime - intime).total_seconds() / 3600.0
            else:
                lived_time = (deathtime - intime).total_seconds() / 3600.0

            # select data within icu stay
            for k in range(len(adm_data)):
                kth_feature = []
                for t, value in adm_data[k]:
                    if intime < t < outtime:

                        kth_feature.append((t, value))

                icu_data.append(kth_feature)


            sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate) 

            # start point of observation
            sample_times = list(filter(lambda x: x < (min(los, lived_time) - observ_win), sample_times))


            for t in sample_times:
                sample_data = []
                vital_label = [0]*12
                # get data
                for feature in range(len(icu_data)):
                    f = []
                    for ft,val in icu_data[feature]:
                        if t <= (ft - intime).total_seconds() / 3600.0 < t + observ_win:
                            f.append((ft, val))
                        
                        if feature < 12 and (t + observ_win) <= (ft - intime).total_seconds() / 3600.0 < (t + observ_win + future_time_interval):
                            vital_label[feature] = 1

                    sample_data.append(f)

                adm_icu_id.append((adm2subj_dict[adm_ids[i]], adm_ids[i], icustay_id))
                wbm_data.append(sample_data)
                wbm_label.append(vital_label) 

    print("Number of created samples:", len(wbm_data), len(adm_icu_id), len(wbm_label))
    print("Number of features:", len(wbm_data[0]))
    return adm_icu_id, wbm_data, wbm_label

CHUNK_KEY = {'ONSET': 0, 'CONTROL': 1, 'ON_INTERVENTION': 2, 'WEAN': 3}
def create_interv_pred(data, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict, adm_ids, interv, sample_rate=6, observ_win=6, eps=1e-6, future_time_interval=4, gap_win=6, start=0, end=10000, need_sample=-1):
    adm_icu_id = []
    ip_data = []
    vent_labels = []
    vaso_labels = []

    # go through icu stays
    for i in trange(start, end):
        if need_sample != -1 and len(ip_data) > need_sample:
            break

        adm_data = data[i] # (23 * T_feature)
        if adm_ids[i] not in icu_dict:
            continue
        for icustay_id in icu_dict[adm_ids[i]]:
            icu_data = []
            if pd.isnull(los_dict[str(adm_ids[i])+'_'+str(icustay_id)]):
                continue

            adm_index = (adm2subj_dict[adm_ids[i]], adm_ids[i], icustay_id)
            

            los = 24.0 * los_dict[str(adm_ids[i])+'_'+str(icustay_id)]  # in hours

            deathtime = adm2deathtime_dict[adm_ids[i]]
            intime = icu_dict[adm_ids[i]][icustay_id][0]
            outtime = icu_dict[adm_ids[i]][icustay_id][1]

            if deathtime is None:
                lived_time = (outtime - intime).total_seconds() / 3600.0
            else:
                lived_time = (deathtime - intime).total_seconds() / 3600.0

            # select data within icu stay
            for k in range(len(adm_data)):
                kth_feature = []
                for t, value in adm_data[k]:
                    if intime < t < outtime:

                        kth_feature.append((t, value))

                icu_data.append(kth_feature)


            sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate) # (采样起点，终点，步长)

            # start point of observation
            sample_times = list(filter(lambda x: x < (min(los, lived_time) - observ_win), sample_times))


            for t in sample_times:
                sample_data = []
                
                
                # get data
                for feature in range(len(icu_data)):
                    f = []
                    for ft,val in icu_data[feature]:
                        if t <= (ft - intime).total_seconds() / 3600.0 < t + observ_win:
                            f.append((ft, val))
                    sample_data.append(f)

                vent_label = cal_label(interv['vent'], adm_index, int(t), observ_win, gap_win, future_time_interval)
                vaso_label = cal_label(interv['vaso'], adm_index, int(t), observ_win, gap_win, future_time_interval)

                if vent_label is not None and vaso_label is not None:
                    adm_icu_id.append(adm_index)
                    ip_data.append(sample_data)
                    vent_labels.append(vent_label) 
                    vaso_labels.append(vaso_label) 

    print("Number of created samples:", len(ip_data), len(adm_icu_id), len(vent_labels))
    print("Number of features:", len(ip_data[0]))
    return adm_icu_id, ip_data, vent_labels, vaso_labels

def cal_label(interv, adm_index, t, observ_win, gap_win, future_time_interval):
    if adm_index not in interv:
        return None
        
    y_patient = interv[adm_index].values

    result_window = y_patient[t+observ_win+gap_win : t+observ_win+gap_win+future_time_interval]

    result_window_diff = set(np.diff(result_window))
    gap_window = y_patient[t+observ_win:t+observ_win+gap_win]
    gap_window_diff = set(np.diff(gap_window))


    if 1 in gap_window_diff or -1 in gap_window_diff:
        result = None
    elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 0):
        result = CHUNK_KEY['CONTROL']
    elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 1):
        result = CHUNK_KEY['ON_INTERVENTION']
    elif 1 in result_window_diff: 
        result = CHUNK_KEY['ONSET']
    elif -1 in result_window_diff:
        result = CHUNK_KEY['WEAN']
    else:
        result = None

    return result