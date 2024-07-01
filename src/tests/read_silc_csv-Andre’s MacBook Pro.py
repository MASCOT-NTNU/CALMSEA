import pandas as pd
import time
import numpy as np
import datetime

def time_now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")




def make_random_csv():
    total_time = 2 * 3600 # 2 hours
    t_start = time.time()
    t_now = t_start



    columns = ['particle index', 'major_axis_length', 'minor_axis_length',
       'equivalent_diameter', 'solidity', 'minr', 'minc', 'maxr', 'maxc',
       'probability_oil', 'probability_other', 'probability_bubble',
       'probability_faecal_pellets', 'probability_copepod',
       'probability_diatom_chain', 'probability_oily_gas', 'export name',
       'timestamp', 'saturation']
    
    probablity_columns = columns[9:16]
    other_columns = columns[1:8] + [columns[16]] + [columns[18]]

    series = {column: [] for column in columns}

    frecuency = 100 # Hz
    particle_index = 0

    while time.time() - t_start < total_time:
        t_now = datetime.datetime.now()
        n_new_values = np.random.randint(0,4)
        for i in range(n_new_values):
            prob = np.random.uniform(size=7)
            sum_prob = prob.sum()
            prob = prob / sum_prob
            for j, column in enumerate(probablity_columns):
                series[column].append(prob[j])
            
            
            series["timestamp"].append(t_now)
            series["particle index"].append(particle_index)
            for column in other_columns:
                series[column].append(0)
            particle_index += 1

        

        print("Time elapsed:", time.time() - t_start)
        time.sleep(1/frecuency)

        df = pd.DataFrame(series)
        print(len(df))
        df.to_csv("random_data.csv", index=True)


def read_silc_csv(file_path):
    """
    Read a csv file with the SILC data
    """
    # Load the csv file
    df = pd.read_csv(file_path)

    # Filter the enties that are none 
    df = df[df["export name"] != "not_exported"]

    # Get the time
    time_stamp = df["timestamp"].values

    # Transform the time to seconds
    time_stamp = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in time_stamp]
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
            return_data["T"].append(time_stamp[i])
            return_data["probability_copepod"].append(copopod_data[i])



    if len(return_data["T"]) == 0:
        print(time_now_str(), "[INFO] [AGENT] No probable copopods observed")
    else:
        print(time_now_str(), "[INFO] [AGENT] Found", len(return_data["T"]), "probable copopods")
    return return_data


def merge_data(current_data, silc_data):
    """
    merge the two dictionaries
    """

    t_curr = np.array(current_data["T"])
    t_silc = np.array(silc_data["T"])

    correction_t = 0
    t_silc = t_silc + correction_t

    copepod_count = np.zeros(len(t_curr))

    for i, t in enumerate(t_silc):
        if np.min(np.abs(t_curr - t)) < 2:
            index = np.argmin(np.abs(t_curr - t))
            copepod_count[index] += 1

    current_data["copepod_count"] = copepod_count
    return current_data


# Test merge data
current_data = {"T": np.linspace(50, 100,100)}

silc_data = {"T": np.linspace(0, 100, 1000)}
silc_data = read_silc_csv("src/test-STATS.csv")
t1 = time.time()
silc_data = read_silc_csv("random_data.csv")
t2 = time.time()
print(f"Time to read data: {t2-t1:.2f} seconds")

merged_data = merge_data(current_data, silc_data)
print(merged_data)
    





#rd = read_silc_csv("random_data.csv")
#print(rd)