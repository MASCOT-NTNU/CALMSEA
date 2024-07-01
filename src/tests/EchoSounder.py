import numpy as np
import pandas as pd
import time
import datetime
import pickle


import os

class EchoSounderData:

    def __init__(self, path):
        self.path = path
        self.data = self.load_raw_data()
        self.data = self.load_data()

    def load_raw_data(self):
        loaded_data = {}
        with np.load(self.path) as data:

            files = data.files

            for f in files:
                loaded_data[f] = data[f]

        loaded_data['x_coords'] = loaded_data['x_coords'][0]
        
        for key in loaded_data.keys():
            print(key)
            print(loaded_data[key].shape)

        return loaded_data

    def load_data(self):
        loaded_data = self.data

        # Adding some attributes to the object
        meshgrid = np.array(loaded_data['meshgrid'])    
        x = meshgrid[:,0]
        y = meshgrid[:,1]
        loaded_data['x'] = x
        loaded_data['y'] = y

        # Get the limits of the data
        loaded_data["x_lim"] = [np.min(x), np.max(x)]
        loaded_data["y_lim"] = [np.min(y), np.max(y)]
        z_max = np.max(loaded_data['lower_s_mean'])
        z_min = np.min(loaded_data['upper_s_mean'])
        loaded_data["z_lim"] = [z_min, z_max]

        print("x_lim", loaded_data["x_lim"])
        print("y_lim", loaded_data["y_lim"])
        print("z_lim", loaded_data["z_lim"])

        # Get the mean of the data
        loaded_data["mean"] = self.get_mean()

        # getting the width of the patch
        loaded_data["patch_width"] = self.get_patch_width()

        return loaded_data
    

    def get_mean(self):
        upper_s_mean = self.data['upper_s_mean']
        lower_s_mean = self.data['lower_s_mean']
        mean_upper = np.mean(upper_s_mean, axis=0)
        mean_lower = np.mean(lower_s_mean, axis=0)
        mean = np.mean(np.array([mean_upper, mean_lower]), axis=0)
        return mean
    
    def get_patch_width(self):
        upper_s_mean = self.data['upper_s_mean']
        lower_s_mean = self.data['lower_s_mean']
        mean = self.get_mean()
        
        diff = []
        for i in range(upper_s_mean.shape[0]):
            if np.abs(upper_s_mean[i] - mean) > 0.1 and np.abs(lower_s_mean[i] - mean) > 0.1:
                diff.append(np.abs(upper_s_mean[i] - lower_s_mean[i]))
        patch_width = np.mean(diff)
        return patch_width
    
    def get_beta0_from_width(self, width):
        return 1 / ((width/2) ** 2)

    def is_s_inside_area(self, s):
        # checks if the point is inside the patch
        x = s[0]
        y = s[1]
        z = s[2]
        if x < self.data["x_lim"][0] or x > self.data["x_lim"][1]:
            return False
        if y < self.data["y_lim"][0] or y > self.data["y_lim"][1]:
            return False
        if z < self.data["z_lim"][0] or z > self.data["z_lim"][1]:
            return False
        return True

    def get_closest_index(self, s):
        # Get the closest index to the point s
        x = s[0]
        y = s[1]

        x_coords = self.data['x_coords']
        y_coords = self.data['y_coords']

        x_index = np.argmin(np.abs(x_coords - x))
        y_index = np.argmin(np.abs(y_coords - y))

        return x_index, y_index

        

    def get_value_S(self, S):
        # This function returns 1 if s is inside the patch and 0 otherwise
        
        values = np.zeros(len(S))

        upper_s_mean = self.data['upper_s_mean']
        lower_s_mean = self.data['lower_s_mean']
        for i, s in enumerate(S):
            if self.is_s_inside_area(s):
                x_ind, y_ind = self.get_closest_index(s)
                ind_mesh = x_ind * len(self.data['y_coords']) + y_ind
                if s[2] > upper_s_mean[ind_mesh] and s[2] < lower_s_mean[ind_mesh]:
                    values[i] = 1
                    
        return values





if __name__ == "__main__":  

    file_path = 'src/echosounder_files/prior_latest.npz'

    echo_sounder_data = EchoSounderData(file_path)

    beta_0 = echo_sounder_data.get_beta0_from_width(echo_sounder_data.data["patch_width"])
    print("beta_0", beta_0)

    # Chech if som random points is inside the patch
    x = np.random.uniform(-200, 2200, 100)
    y = np.random.uniform(-200, 2200, 100)
    z = np.random.uniform(10, 40, 100)

    for i in range(100):
        s = np.array([x[i], y[i], z[i]])
        #print(s, echo_sounder_data.is_s_inside_area(s))


    # Chech if som random points is inside the patch
    n = 5000
    x = np.random.uniform(-200, 2200, n)
    y = np.random.uniform(-200, 2200, n)
    z = np.random.uniform(10, 40, n)
    S = np.array([x.T, y.T, z.T]).T
    inside = echo_sounder_data.get_value_S(S)
    print("inside", inside)

    # 3D scatter plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, v in enumerate(inside):
        if v == 1:
            ax.scatter(x[i], y[i], z[i], c='r')
        else:
            pass 
            #ax.scatter(x[i], y[i], z[i], c='b')
    plt.show()

