import numpy as np
import pandas as pd
import time 

while True:
    # read the csv file
    df = pd.read_csv("random_data.csv")
    print(len(df))
    time.sleep(2)   