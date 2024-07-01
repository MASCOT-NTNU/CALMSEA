import numpy as np
import pandas as pd
import datetime
import time


def time_now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


csv_name = "random_data.csv"
new_csv_name = "random_data_processed.csv"
sleep_time = 2

"""
This is reducing the size of the csv file by only keeping the entries that have a probability of a copepod larger than 0.5
This makes it a bit faster to send to the main cpu
"""

while True:

    try:

        # Read the csv file

        df = pd.read_csv(csv_name)

        len_old = len(df)

        # Filter the enties that are none
        df = df[df["export name"] != "not_exported"]

        # Filter where the probability is larger than 0.5
        df = df[df["probability_copepod"] > 0.5]

        # Only keep the columns that are needed
        df = df[["timestamp", "probability_copepod"]]

        len_new = len(df)
        # Save the new csv file
        df.to_csv(new_csv_name, index=False)

        print(time_now_str(), f"pd len reduced from {len_old} too {len_new}")

        time.sleep(sleep_time)
    except:
        print(time_now_str(), "Could not process file")
        time.sleep(sleep_time)



