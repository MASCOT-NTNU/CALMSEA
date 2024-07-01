import numpy as np
import pandas as pd
import time
import datetime
import pickle

import os

def time_now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_silc_csv(file_path , threshold = 0.9):
    """
    Read a csv file with the SILC data
    """
    # Load the csv file
    try:
        df = pd.read_csv(file_path)
    except:
        print(time_now_str(), "[ERROR] [AGENT] Could not read the SILC data")
        return {"T": []}

    # Filter the enties that are none 
    df = df[df["export name"] != "not_exported"]

    # Get the time
    time_stamp = df["timestamp"].values

    # Transform the time to seconds
    try:
        time_stamp_temp = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in time_stamp]
    except:
        print(time_now_str(), "[ERROR] [AGENT] Could not convert the time to seconds")

        time_stamp_temp = []
        for x in time_stamp:

            if str(x) == "nan":
                time_stamp_temp.append(time_stamp_temp[-1])
            else: 
                time_stamp_temp.append(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
        
        print(time_stamp)
    
    time_stamp = time_stamp_temp
    time_stamp = [x.timestamp() for x in time_stamp]

    probablity_columns = df.columns[9:16]
    copopod_data = df['probability_copepod'].values
    max_probability_df = df[probablity_columns]

    # for each row get the max probability column
    max_probability = max_probability_df.idxmax(axis=1)

    max_probability = max_probability.values

    columns_considered = ["probability_copepod"]

    return_data = {"T": [], "probability_copepod": []}
    for i, prob in enumerate(max_probability):
        if prob in columns_considered:
            if copopod_data[i] > threshold:
                return_data["T"].append(time_stamp[i])
                return_data["probability_copepod"].append(copopod_data[i])



    if len(return_data["T"]) == 0:
        print(time_now_str(), "[INFO] [AGENT] No probable copopods observed")
    else:
        print(time_now_str(), "[INFO] [AGENT] Found", len(return_data["T"]), "probable copopods")
    return return_data


def merge_data(current_data, silc_data, correction_t=0):
    """
    merge the two dictionaries
    """

    t_curr = np.array(current_data["T"])
    t_silc = np.array(silc_data["T"])

    if len(t_silc) == 0:
        if "copepod_count" in current_data.keys():
            diff_len = len(t_curr) - len(current_data["copepod_count"])
            current_data["copepod_count"] = np.concatenate([current_data["copepod_count"], np.zeros(diff_len)])
        else:
            current_data["copepod_count"] = np.zeros(len(t_curr))

        return current_data

    t_silc = t_silc + correction_t

    copepod_count = np.zeros(len(t_curr))

    for i, t in enumerate(t_silc):
        if np.min(np.abs(t_curr - t)) < 2:
            index = np.argmin(np.abs(t_curr - t))
            copepod_count[index] += 1

    current_data["copepod_count"] = copepod_count
    return current_data