import numpy as np
import pandas as pd
import time
import datetime
import pickle
import matplotlib.pyplot as plt


import os

from help_func_postprocess import *

silc_data_path = "src/silc_data/"
silc_data_name = "RAW-STATS-testdive-3.csv"
silc_data = read_silc_csv(silc_data_path + silc_data_name)
silc_df = pd.DataFrame(silc_data)

df_columns = silc_df.columns
print(df_columns)

plt.scatter(silc_df["T"], silc_df["probability_copepod"])
plt.show()