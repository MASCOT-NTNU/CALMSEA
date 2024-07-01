
import numpy as np

time_remaining = 10000
hrs = int(np.floor(time_remaining / 3600))
mins = int(np.floor((time_remaining - hrs * 3600) / 60))
print(hrs, mins)
print(time_remaining / 3600)