import sys
 
# adding src to the path to import the module
sys.path.insert(0, '/Users/ajolaise/Library/CloudStorage/OneDrive-NTNU/PhD/code/2023/PoissonModel3D/src/')

from PathPlanner import PathPlanner
from Field import Field
from Model import Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches
import seaborn as sns
import time
import random



print("This is the RunSimulation class")
print("This class is used to run the simulation")

# Parameters , to set up the simulation we need to define the parameters

random_setup = False
print_while_running = True
plot_simulation = False

# Path Planning parameters
n_directions = 8
n_depths = 9
max_planning_time = 60
max_speed = 1.0 # m/s
attack_angle = 10 # degrees
sampling_frequency = 1/10 # Hz 
yoyo_step_length = 500 # m
step_length = 100 # m

# Field parameters
s_lim = [[0,2000], [0,2000], [0,50]]
t_lim = [0, 24*3600]
n_s = [10,11,12]
n_t = 14
phi_spatial_z = 1 / 4
phi_spatial_yx = 1 / 200
phi_temporal = 1 / 3600
sigma = 2
Beta = (0.1, 3, 2, 2/ 6**2)

# Model parameters
Beta_model = Beta
phi_z_model = phi_spatial_z
phi_yx_model = phi_spatial_yx
phi_t_model = phi_temporal
sigma_model = sigma

# Objective function 
objective_function = "max_expected_improvement"

# Simulation parameters


# Create the field
print(" Creating the field ")
t1 = time.time()
field = Field(phi_spatial_z=phi_spatial_z,
                phi_spatial_yx=phi_spatial_yx,
                phi_temporal=phi_temporal, 
                sigma=sigma, 
                s_lim=s_lim, 
                t_lim=t_lim, 
                n_s=n_s, 
                n_t=n_t,
                generate_field_x=True,
                print_while_running=print_while_running)

t2 = time.time()
print("Time to create the field: ", round(t2 - t1,2), " seconds")

# Create the model
print(" Creating the model ")
def get_intensity_mu_x(S: np.ndarray,T: np.ndarray) -> np.ndarray:
    beta_0 = 0.5 # The mean intensity
    beta_1 = 2 # The amplitude of the intensity
    beta_2 = 1 / 6**2 # related to the width of the intensity
    peak_depth_min = 10
    peak_depth_max = 60

    Sz = S[:,2]

    # The intesity function is dependant on the depth and time of day 
    phase = 2 * 3.1415926 * T /(24 * 3600)
    peak_depth = np.sin(phase) * (peak_depth_max - peak_depth_min) + peak_depth_min
    return np.exp(-(peak_depth - Sz)**2 * beta_2) * beta_1 + beta_0

model = Model(prior_mean_function=get_intensity_mu_x, 
                phi_z=phi_z_model,
                phi_yx=phi_yx_model,
                phi_temporal=phi_t_model,
                sigma=sigma_model,
                print_while_running=print_while_running)


# Create the path planner
print(" Creating the path planner ")
path_planner = PathPlanner(field_limits=s_lim, print_while_running=False,
                            max_speed=max_speed, 
                            attack_angle=attack_angle,
                            sampling_frequency=sampling_frequency,
                            yoyo_step_length=yoyo_step_length,
                            step_length=step_length)


# Add the first observations at the surface
m = 7
sx_start = np.repeat(np.random.uniform(s_lim[0][0], s_lim[0][1]), m)
sy_start = np.repeat(np.random.uniform(s_lim[1][0], s_lim[1][1]), m)
sz_start = np.repeat(0, m)
sx_start = sx_start + np.random.normal(0, 1, m)
sy_start = sy_start + np.random.normal(0, 1, m)
sz_start = sz_start + np.abs(np.random.normal(0, 1, m))
s_start_samples = np.array([sx_start, sy_start, sz_start]).T
t_start_samples = np.linspace(0, 10 * m, m)
y_start = field.get_observation_Y(s_start_samples, t_start_samples)
model.add_new_values(s_start_samples, t_start_samples, y_start)


def guessing_method(s_old, s_new,x_old, y_old, y_obs,mu_new, method="flat"):

    if method == "average":
        return np.repeat(np.mean(x_old), len(y_obs))
    
    if method == "close_average":
        return np.repeat(np.mean(x_old[-5:]), len(y_obs))
    
    if method == "sliding_average":
        n_new = len(y_obs)
        y_obs_log = np.log(y_obs + 1)
        y_all = np.concatenate((x_old, y_obs_log))
        window_size = 5
        average = np.convolve(y_all, np.ones(window_size), 'valid') / window_size
        if average.shape[0] < n_new:
            return average[-1] * np.ones(n_new)
        if average.shape[0] == n_new:
            return average
        # return the last n_new values
        return average[-n_new:]
    
    if method == "log_observation":
        return np.log(y_obs + 1)
    
    if method == "prior":
        return mu_new


methods = ["close_average", "sliding_average", "log_observation", "average","prior"]






        






# Dataset 
n_old_list = []
n_new_list = []
len_norm_list = []
tol_list = []
error_new = [] 
error_list = [] # This the total error in the fitted model
error_sd_list = [] # This the total error in the fitted model
norm_list_list  = []
guess_type = []
timing_list = []





# How many steps to take at a time 
m_range = 10
k = 50
for i in range(k):  

    S_new = None
    T_new = None
    Y_new = None

    # Get the current location and time
    s_now = model.data_dict["S"][-1]
    t_now = model.data_dict["T"][-1]
    
    m = np.random.randint(1,m_range)

    # Adding multiple steps at the time 
    for step in range(m):

        # Random next location
        s_next = np.array([np.random.uniform(s_lim[0][0], s_lim[0][1]), np.random.uniform(s_lim[1][0], s_lim[1][1]), np.random.uniform(s_lim[2][0], s_lim[2][1])])
        while np.linalg.norm(s_next[:2] - s_now[:2]) < 400:
            s_next = np.array([np.random.uniform(s_lim[0][0], s_lim[0][1]), np.random.uniform(s_lim[1][0], s_lim[1][1]), np.random.uniform(s_lim[2][0], s_lim[2][1])])
        
        # Suggest next location
        paths = path_planner.get_possible_paths_z(s_now, s_next, [s_lim[2][0], s_lim[2][1]])
        path = paths[np.random.randint(0, len(paths))]

        # Get the next observations
        S_new_temp = path["S"]
        T_new_temp = path["T"]
        Y_new_temp = field.get_observation_Y(path["S"], path["T"])

        # Add the new values
        if S_new is not None:
            S_new = np.concatenate((S_new, S_new_temp))
            T_new = np.concatenate((T_new, T_new_temp))
            Y_new = np.concatenate((Y_new, Y_new_temp))
        else:
            S_new = S_new_temp
            T_new = T_new_temp
            Y_new = Y_new_temp

        # Setting the s and t now
        s_now = S_new[-1]
        t_now = T_new[-1]

    # Get the true intensity
    x_true = field.get_intensity_x(path["S"], path["T"])
    




    # Load data from memory
    s_old = model.data_dict["S"]
    t_old = model.data_dict["T"]
    y_old = model.data_dict["y"]
    mu_old = model.data_dict["mu"]
    Sigma_old = model.data_dict["Sigma"]
    x_old = model.data_dict["x"]
    P_old = model.data_dict["P"]


    # Get the dimensions of the old and new data
    n_old, n_new = len(y_old), len(Y_new)


    # Join the old and new data
    s = np.concatenate((s_old, S_new))
    t = np.concatenate((t_old, T_new))
    y = np.concatenate((y_old, Y_new))
    mu_new = model.prior_mean_function(S_new, T_new) 
    mu = np.concatenate((mu_old, mu_new))

    x_true_all = field.get_intensity_x(s, t)

    # Get the cross covariance matrix
    # This can be done more effieciently
    Sigma = model.make_covariance_matrix(s, t)


    

    # Fit the model 
    tol = 1
    # shuffle the methods
    random.shuffle(methods)
    for method in methods:

        print(f"###### method: {method} , k: {i} ######")

        # Use the five last values in the previouse estimate of x
        new_guess = guessing_method(s_old, S_new,x_old, y_old, Y_new,mu_new, method=method)
        x_init = np.concatenate((x_old, new_guess))
        Sigma = model.make_covariance_matrix(s, t)
        t1 = time.time()
        
        x_fitted, Sigma_est, P, norm_list = model.fit(x_init, mu, y, Sigma, tol=tol)
        t2 = time.time()

        # Update the lists
        guess_type.append(method)
        n_old_list.append(n_old)
        n_new_list.append(n_new)
        len_norm_list.append(len(norm_list))
        tol_list.append(tol)
        abs_error = np.sum(np.abs(x_fitted - x_true_all))
        newest_error = np.sum(np.abs(x_fitted - x_true_all)[-n_new:])
        error_new.append(newest_error)
        error_list.append(np.sum(np.abs(x_fitted - x_true_all)))
        error_sd_list.append(np.std(x_fitted - x_true_all))
        norm_list_list.append(norm_list)
        timing_list.append(t2-t1)

        plt.scatter(np.arange(len(x_true_all)), x_true_all, label="True model")
        plt.plot(x_fitted, label="Fitted model")
        plt.plot(x_init, label="Initial guess")
        plt.plot(mu, label="Prior mean")
        plt.legend()
        plt.title(f"Method: {method}, iterations: {len(norm_list)}")
        plt.savefig(f"figures/fitting_model/plot_fitting_{i}_{method}.png")
        plt.close()

   

    


    # Add the new values to the model
    print("############ Adding the values to the model #####" )
    t1 = time.time()
    model.add_new_values(S_new, T_new, Y_new)
    t2 = time.time()
    if t2 - t1 > 1:
        print("Time to add the values to the model: ", round(t2 - t1,2), " seconds")
        model.down_sample_points()

    if len(t) > 2000:
        break

for norm_list in norm_list_list:
    plt.plot(np.flip(norm_list))


# Make the x-axis logarithmic
plt.xscale("log")
plt.yscale("log")
plt.savefig("figures/fitting_model/norm_list.png")
plt.close()

# Make a dataframe
df = pd.DataFrame({"n_old": n_old_list, 
                   "n_new": n_new_list, 
                   "len_norm": len_norm_list, 
                   "tol": tol_list, 
                   "error": error_list,
                   "error_new": error_new,
                   "error_sd": error_sd_list,
                   "time": timing_list,
                   "method": guess_type})


df.to_csv("data/fitting_model_test/fitting_model.csv")

levels, categories = pd.factorize(df['method'])
colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

plt.scatter(df["n_old"] + df["n_new"], df["error"],  c=colors)
plt.xlabel("# points in model + # points added")
plt.ylabel("absolute error fitted model")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/error_absolute.png")
plt.close()

plt.scatter(df["n_old"] + df["n_new"], df["error_sd"], c=colors)
plt.xlabel("# points in model + # points added")
plt.ylabel("Sd")
plt.title("Error between the fitted and true x")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/error_sd.png")
plt.close()

plt.scatter(df["n_old"] + df["n_new"], df["len_norm"], c=colors)
plt.xlabel("# points in model + # points added")
plt.ylabel("Iterations to fit model")
plt.title("Iterations to fit the model vs total points")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/itter_fit_model_opn.png")
plt.close()

plt.scatter(df["n_new"], df["len_norm"], c=colors)
plt.xlabel("points added")
plt.ylabel("Iterations to fit model")
plt.title("Iterations to fit the model vs new points")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/itter_fit_model_n.png")
plt.close()

plt.scatter(df["n_old"], df["len_norm"],  c=colors)
plt.xlabel("# points in model")
plt.ylabel("Iterations to fit model")
plt.title("Iterations to fit the model vs old points")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/itter_fit_model_o.png")
plt.close()

time_pr_iteration = df["time"] / df["len_norm"]

plt.scatter(df["n_old"] + df["n_new"], time_pr_iteration,  c=colors)
plt.ylabel("Time pr iteration (s)")
plt.xlabel("# points in model + # points added")
plt.title("Time pr iteration")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/time_pr_iteration.png")
plt.close()


# time to fit the model
plt.scatter(df["n_old"] + df["n_new"], df["time"],  c=colors)
plt.ylabel("Time (s)")
plt.xlabel("# points in model + # points added")
plt.title("Time to fit the model")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/time_fit_model.png")
plt.close()


# Plot multiple boxplots on one graph
fig, ax = plt.subplots(3,3, figsize=(15,15))

sns.boxenplot(x="method", y="error", data=df, ax=ax[0,0])
sns.boxenplot(x="method", y="error_sd", data=df, ax=ax[0,1])
sns.boxenplot(x="method", y="len_norm", data=df, ax=ax[0,2])
sns.boxenplot(x="method", y="time", data=df, ax=ax[1,0])
sns.boxenplot(x="method", y="n_old", data=df, ax=ax[1,1])
sns.boxenplot(x="method", y="n_new", data=df, ax=ax[1,2])
#sns.boxenplot(x="method", y="n_old" + df["n_new"], data=df, ax=ax[2,0])
#sns.boxenplot(x="method", y="time" / df["len_norm"], data=df, ax=ax[2,1])
sns.boxenplot(x="method", y="time", data=df, ax=ax[2,2])

plt.savefig("figures/fitting_model/boxplot.png")
plt.close()