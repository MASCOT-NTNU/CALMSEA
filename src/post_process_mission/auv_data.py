import numpy as np
import pandas as pd
import time
import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from help_func_postprocess import *

import os


path = "src/mission_data/mission_data/"
silc_data_path = "src/silc_data/"
mission = "agent_cla_mausund_20240606_173552"
mission = "agent_cla_mausund_20240529_172941"

missions = [
    {"data_folder": "agent_cla_mausund_20240606_173552",
     "silc_data_name": "RAW-STATS-cla_0606.csv",
     "time_correction": 2 * 60*60},
    {"data_folder": "agent_cla_mausund_20240529_172941",
     "silc_data_name": "RAW-STATS-cla_0607.csv",
     "time_correction": 0},
     {"data_folder": "agent_silcam_trondheim_20240605_141749",
      "silc_data_name": "RAW-STATS_mausund_run1.csv",
      "time_correction": 60*60},
    {"data_folder": "agent_silcam_trondheim_20240605_153800",
     "silc_data_name": "RAW-STATS_mausund_run2.csv",
     "time_correction": 60*60}      
]






# Get all the files in the folder
files = os.listdir(path + mission)
print(files)


for mission in missions:
    mission_files = os.listdir(path + mission["data_folder"])
    mission["files"] = mission_files
    iterations = []

    for file in mission_files:
        counter = int(file.split("_")[-1].split(".")[0])
        if counter not in iterations:
            iterations.append(counter)

    # Sort the iterations
    iterations = np.array(iterations, dtype=int)
    mission["iterations"] = np.sort(iterations)


    # Create folder to store the plots
    plot_path = "figures/post_processing/mission_data/" + mission["data_folder"]
    if not os.path.exists(plot_path):
        os.makedirs(plot_path) 


    file_types = ["data", "model_data", "parameters"]

    # Get the last data file
    last_file = "data_" + str(mission["iterations"][-1]) + ".pkl"
    last_file_load_path = path + mission["data_folder"] + "/" + last_file   

    # Load the data
    data = pickle.load(open(last_file_load_path, "rb"))

    # Load the SILC data
    silc_data = read_silc_csv(silc_data_path + mission["silc_data_name"])
    silc_t = silc_data["T"]
    data_t = data["T"]

    print("SILC data: ", silc_t[0], silc_t[-1])
    print("Data: ", data_t[0], data_t[-1])
    # Merge the data
    data = merge_data(data, silc_data, mission["time_correction"])
    print(data["copepod_count"])


    # Load the data 
    depth = np.array(data["depth"])
    S = np.array(data["S"])
    x = S[:, 1]
    y = S[:, 0]
    T = np.array(data["T"])
    mission_duration = T - T[0]
    mission_duration_hours = mission_duration / (60*60)
    depth = np.array(data["depth"])
    salinity = np.array(data["salinity"])
    temperature = np.array(data["temperature"])
    chlorophyll = np.array(data["chlorophyll"])
    copepod_count = np.array(data["copepod_count"])




    # plot the data
    plt.figure()
    plt.plot(T, depth, label="Depth")
    plt.plot(T, salinity, label="Salinity")
    plt.plot(T, temperature, label="Temperature")
    plt.plot(T, chlorophyll, label="Chlorophyll")
    plt.legend()
    plt.savefig(plot_path + "/data.png")
    plt.close()



    # Plot the path of the AUV
    plt.figure()
    plt.scatter(x, y, c=depth)
    plt.colorbar()
    plt.savefig(plot_path + "/auv_path.png")
    plt.close()

    # Plot the path of the AUV time colored
    plt.figure()
    plt.scatter(x, y, c=mission_duration_hours)
    plt.colorbar()
    plt.savefig(plot_path + "/auv_path_time.png")
    plt.close()

    


    plt.figure()
    plt.plot(T, depth, label="Depth")
    plt.plot(T, chlorophyll, label="Chlorophyll")
    plt.legend()
    plt.savefig(plot_path + "/data_depth_chlorophyll.png")
    plt.close()


    plt.figure()
    plt.scatter(mission_duration_hours, depth,c=chlorophyll,  label="Depth",
                cmap='viridis', alpha=0.6, vmin = 0, vmax = 4)
    plt.legend()
    plt.colorbar()
    # Flip the y-axis
    plt.gca().invert_yaxis()
    plt.ylabel("Depth (m)")
    plt.xlabel("Time (hrs)")
    plt.title("Cholorophyll")
    plt.savefig(plot_path + "/data_chlorophyll.png")
    plt.close()

    plt.figure()
    plt.scatter(depth, chlorophyll)
    plt.xlabel("Depth")
    plt.ylabel("Chlorophyll")
    plt.savefig(plot_path + "/scatter_depth_chlorophyll.png")
    plt.close()


    # Plot AUV path 
    plt.figure()
    plt.scatter(x, y, c=chlorophyll)
    plt.colorbar()
    plt.savefig(plot_path + "/auv_path_chlorophyll.png")
    plt.close()


    # Plot copepod count vs depth
    plt.figure()
    corr = np.corrcoef(depth, data["copepod_count"])
    plt.title("Correlation: " + str(corr[0, 1]))
    plt.scatter(depth, data["copepod_count"])
    plt.xlabel("Depth")
    plt.ylabel("Copepod count")
    plt.savefig(plot_path + "/scatter_depth_copepod_count.png")
    plt.close()

    # Plot copepod count vs chlorophyll

    corr = np.corrcoef(chlorophyll, data["copepod_count"])
    plt.figure()
    plt.title("Correlation: " + str(corr[0, 1]))
    plt.scatter(chlorophyll, data["copepod_count"])
    plt.xlabel("Chlorophyll")
    plt.ylabel("Copepod count")
    plt.savefig(plot_path + "/scatter_chlorophyll_copepod_count.png")
    plt.close()


    # plot desity lines for depth and chlorophyll
    plt.figure()
    plt.hist(depth, bins=20, density=True, alpha=0.6, color='b')

    

