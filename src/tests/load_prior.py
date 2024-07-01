import numpy as np


# file path
file_path = 'src/echosounder_files/prior 1.npz'

# Load the arrays
with np.load(file_path) as data:
    print(data)
    loaded_array1 = data['upper_s_mean']
    loaded_array2 = data['lower_s_mean']

    loaded_array2 = data['heat_map']



import matplotlib.pyplot as plt

print(loaded_array1)
print(loaded_array2)
