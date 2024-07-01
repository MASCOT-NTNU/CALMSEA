import numpy as np
import pandas as pd
import time

total_time = 2 * 3600 # 2 hours
t_start = time.time()
t_now = t_start

time_series = []
value_series = []

frecuency = 2 # Hz

while t_now - t_start < total_time:
    t_now = time.time()
    n_new_values = np.random.randint(0,4)
    for i in range(n_new_values):
        time_series.append(t_now)
        value_series.append(np.random.uniform())
    print("Time elapsed:", t_now - t_start)
    time.sleep(1/frecuency)

    df = pd.DataFrame({"time": time_series, "value": value_series})
    df.to_csv("random_data.csv", index=True)