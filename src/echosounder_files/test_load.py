import numpy as np


# file path
file_path = 'prior.npz'

# Load the arrays
with np.load(file_path) as data:
    print(data)
    loaded_array1 = data['upper_s_mean']
    loaded_array2 = data['lower_s_mean']

    loaded_array2 = data['heat_map']