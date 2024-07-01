import numpy as np
import pandas as pd
import time
import datetime
import pickle
import matplotlib.pyplot as plt


import os

from help_func_postprocess import *


path = "src/mission_data/mission_data/"

mission = "agent_silcam_trondheim_20240605_141749"
mission = "agent_silcam_trondheim_20240605_153800"
silc_data_name = "RAW-STATS_mausund_run1.csv"
silc_data_name = "RAW-STATS_mausund_run2.csv"
#mission = "agent_silcam_trondheim_20240528_115742"

# Create folder to store the plots
plot_path = "figures/post_processing/mission_data/" + mission 
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


# Get all the files in the folder
files = os.listdir(path + mission)
print(files)

file = "data_15.pkl"
file_types = ["data", "model_data", "parameters"]

# Load the data
data = pickle.load(open(path + mission + "/" + file, "rb"))

print(data["copepod_count"])


chlorophyll = np.array(data["chlorophyll"])
depth = np.array(data["depth"])

# Filter depth
not_surface_ids = np.where(depth > 1)
depth = depth[not_surface_ids]
chlorophyll = chlorophyll[not_surface_ids]

# Get the average chlorophyll
average_chlorophyll = np.mean(chlorophyll)
print("Average chlorophyll: ", average_chlorophyll)
# Get the log of the average chlorophyll
log_average_chlorophyll = np.log(average_chlorophyll)
print("Log average chlorophyll: ", log_average_chlorophyll)
# get the top 1 % of the chlorophyll
top_1_chlorophyll = np.percentile(chlorophyll, 99)
print("Top 1 % chlorophyll: ", top_1_chlorophyll)
# get the log of the top 1 % of the chlorophyll
log_top_1_chlorophyll = np.log(top_1_chlorophyll)
print("Log top 1 % chlorophyll: ", log_top_1_chlorophyll)


silc_data_path = "src/silc_data/"

silc_data = read_silc_csv(silc_data_path + silc_data_name)
silc_df = pd.DataFrame(silc_data)

t_correct = 60*60 # Time correction
data = merge_data(data, silc_data, t_correct)

print(data["copepod_count"])



print(data.keys())
for key in data.keys():
    print(key)

# Load the data 
depth = data["depth"]
S = np.array(data["S"])
T = np.array(data["T"])
x = S[:, 1]
y = S[:, 0]
depth = data["depth"]
salinity = data["salinity"]
temperature = data["temperature"]
chlorophyll = data["chlorophyll"]
copepod_count = data["copepod_count"]
#copepod_count = data["Y"]

# plot the data

plt.figure()
plt.plot(T, depth, label="Depth")
plt.plot(T, salinity, label="Salinity")
plt.plot(T, temperature, label="Temperature")
plt.plot(T, chlorophyll, label="Chlorophyll")
plt.plot(T, copepod_count, label="Copepod count")
plt.legend()
plt.savefig(plot_path + "/data_series.png")
plt.close()


plt.figure()
plt.plot(T, depth, label="Depth")
plt.plot(T, chlorophyll, label="Chlorophyll")
plt.legend()
plt.savefig(plot_path + "/depth_chlorophyll.png")
plt.close()

plt.figure()
depth = np.array(depth)
chlorophyll = np.array(chlorophyll)
not_surface_ids = np.where(depth > 1)
plt.scatter(depth[not_surface_ids], chlorophyll[not_surface_ids])
corr = np.corrcoef(depth[not_surface_ids], chlorophyll[not_surface_ids])
plt.title("Correlation: " + str(corr[0, 1]))
plt.xlabel("Depth")
plt.ylabel("Chlorophyll")
plt.savefig(plot_path + "/depth_chlorophyll_scatter.png")
plt.close()


corr = np.corrcoef(depth[not_surface_ids], copepod_count[not_surface_ids])
plt.figure()
plt.scatter(depth[not_surface_ids], copepod_count[not_surface_ids])
plt.xlabel("Depth")
plt.ylabel("Copepod count")
plt.title("Correlation: " + str(corr[0, 1]))
plt.savefig(plot_path + "/depth_copepod_count_scatter.png")
plt.close()

plt.figure()
plt.scatter(chlorophyll, copepod_count)
plt.xlabel("Chlorophyll")
plt.ylabel("Copepod count")
plt.savefig(plot_path + "/chlorophyll_copepod_count_scatter.png")
plt.close()


# Plotting the path of the AUV
plt.figure()
plt.scatter(x, y, c=chlorophyll, cmap='viridis')
plt.colorbar()
plt.savefig(plot_path + "/auv_path.png")
plt.close()






for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
    print("Threshold: ", threshold)
    silc_data = read_silc_csv(silc_data_path + silc_data_name)
    silc_df = pd.DataFrame(silc_data)

    t_correct = 60*60
    print(silc_data.keys())
    silc_data["T"] = np.array(silc_data["T"])
    silc_data["probability_copepod"] = np.array(silc_data["probability_copepod"])

    # Filter the data by the threshold
    indecies = np.where(silc_data["probability_copepod"] > threshold)
    silc_data["probability_copepod"] = silc_data["probability_copepod"][indecies]
    silc_data["T"] = silc_data["T"][indecies]
    data_merged = merge_data(data, silc_data, t_correct)

    depth = np.array(data_merged["depth"])
    copepod_count = np.array(data_merged["copepod_count"])

    # filter depth
    indecies = np.where(depth > 1)

    depth = depth[indecies]
    copepod_count = copepod_count[indecies]

    # Get correlation
    correlation = np.corrcoef(depth, copepod_count)
    print("Correlation: ", correlation[0, 1])

    plt.figure()
    plt.scatter(depth, copepod_count)
    plt.xlabel("Depth")
    plt.ylabel("Copepod count")
    plt.title("Threshold: " + str(threshold) + " Correlation: " + str(correlation[0, 1]))
    plt.legend()
    plt.savefig(plot_path + "/depth_copepod_count_scatter_threshold_" + str(threshold) + ".png")
    plt.close()