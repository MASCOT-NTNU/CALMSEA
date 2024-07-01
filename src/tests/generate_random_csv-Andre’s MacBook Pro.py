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

    do_print = True

    

    columns = ['particle index', 'major_axis_length', 'minor_axis_length',
       'equivalent_diameter', 'solidity', 'minr', 'minc', 'maxr', 'maxc',
       'probability_oil', 'probability_other', 'probability_bubble',
       'probability_faecal_pellets', 'probability_copepod',
       'probability_diatom_chain', 'probability_oily_gas', 'export name',
       'timestamp', 'saturation']
    
    probablity_columns = columns[9:16]
    other_columns = columns[1:9] + [columns[16]] + [columns[18]]

    series = {column: [] for column in columns}

    frecuency = 2 # Hz
    particle_index = 0

    while time.time() - t_start < total_time:
        t_now = datetime.datetime.now()
        n_new_values = np.random.randint(0,20)
        for i in range(n_new_values):
            prob = np.random.uniform(size=7)
            sum_prob = prob.sum()
            prob = prob / sum_prob
            for j, column in enumerate(probablity_columns):
                if column != 'probability_copepod':
                    series[column].append(prob[j])

            series['probability_copepod'].append(np.random.uniform())
            
            
            
            series["timestamp"].append(t_now)
            series["particle index"].append(particle_index)
            for column in other_columns:
                series[column].append(0)
            particle_index += 1

        
        if do_print:
            print(time_now_str(), "Time elapsed:", time.time() - t_start, "len:", len(series["particle index"]))
        time.sleep(1/frecuency)

        if particle_index % 1000 == 0:
            print(particle_index)

        df = pd.DataFrame(series)
        df.to_csv("/mnt/DATA/proc/random_data.csv", index=False)

make_random_csv()