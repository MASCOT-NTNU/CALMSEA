from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import imageio

from Field import Field
from Model import Model
from ObjectiveFunction import ObjectiveFunction
from PathPlanner import PathPlanner
from help_func import prior_function, get_possible_next_locations, plot_timing

# parameters for the field







# Create a field
field = Field()

# Generate the field
t_lim = [0, 24*3600] # The time range in seconds eq 24 hours
s_lim = [0, 50] # The depth range in m
n_s = 50 # The number of points in the depth range
n_t = 50 # The number of points in the time range
field.generate_field_x(s_lim, t_lim, n_s, n_t)

# Create a model
model = Model(prior_mean_function=prior_function, print_while_running=True)
objective_function = ObjectiveFunction("max_expected_improvement")


# Paramers for the simulation
t_start = 0
t_now = t_start
t_end = (24 * 3600-100) -2000
max_planning_time = 0.5 # seconds

diving_speed = 5 / 60 # m/s, 5 m/min 
sampling_frequency = 1/60 # Hz , one sample per minute

phi_temporal = 1 / 3600 # 1/s, the temporal decay of the observations


# Add the first observation at the surface
s_start_samples = np.array([0,1,2,3])
t_start_samples = np.array([0,60,120,180]) + t_start
y_start = field.get_observation_Y(s_start_samples, t_start_samples)
model.add_new_values(s_start_samples, t_start_samples, y_start)

# Plot simulation
plot_simulation = True

# simulate an adaptive sampling strategy
iter_i = 0
while t_now < t_end and iter_i < 1000:
    iter_i += 1
    print("====== i =======", iter_i)

    t_alg_start = time.time()

    # Get the current location and time
    s_now = model.data_dict["S"][-1]
    t_now = model.data_dict["T"][-1]

    # Find the possible next locations
    s_next, t_next = get_possible_next_locations(s_now, t_now, diving_speed, sampling_frequency, s_lim)

    #print("possible next locations", s_next) # REMOVE
    # Predict the intensity at the possible next locations
    predictions = []
    #print("s_now", s_now, "t_now", t_now) # REMOVE
    for i in range(len(s_next)):
        # Predict the intensity at the next location
        #print("s", s_next[i], "t", t_next[i])
        s_pred = np.repeat(s_next[i], 2)
        t_pred = np.array([t_next[i], t_next[i]+1])
        mu_pred, Sigma_pred = model.predict(s_pred, t_pred)
        loc_dict = {"s": s_next[i], "t": t_next[i], "x_pred": mu_pred, "Sigma_pred": Sigma_pred}
        predictions.append(loc_dict)

    # Use the objective function to find the next design
    next_location = objective_function.objective_function(model, predictions)


    # Get the observation at the next location
    next_s = np.repeat(next_location["s"], 3)
    next_t = np.array([next_location["t"], next_location["t"]+1, next_location["t"]+2])
    y_next = field.get_observation_Y(next_s, next_t)

    
    # Add the observation to the model
    model.add_new_values(next_s, next_t, y_next)

    t_alg_end = time.time()

    # Plot the simulation
    if plot_simulation:
        # Plot the depth profile at the current time
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        t_str = str(round(t_now / (3600),2))
        fig.suptitle("Depth profile at time " + t_str + " hrs,  i = " + str(iter_i))

        # Plot the field
        s_porfile = np.linspace(s_lim[0], s_lim[1], 50)
        t_profile = np.repeat(t_now, len(s_porfile))
        intensity_now = field.get_intensity_x(s_porfile, t_profile)

        # Predict the intensity at the current time
        mu_pred, Sigma_pred = model.predict(s_porfile, t_profile)  
        """
        ax[0].plot(mu_pred,s_porfile, label="Predicted intensity")
        ax[0].plot( intensity_now,s_porfile, label="True intensity")
        ax[0].fill_between( mu_pred - 2 * np.sqrt(np.diag(Sigma_pred)), mu_pred + 2 * np.sqrt(np.diag(Sigma_pred)),s_porfile, alpha=0.2)
        ax[0].legend()
        ax[0].xlabel("Intensity")
        """
        # Plot the observations
        s_obs = model.data_dict["S"]
        t_obs = model.data_dict["T"]
        y_obs = model.data_dict["y"]

        # Fade out points based on how long ago they were observed
        t_dist = np.abs(t_obs - t_now)
        t_alpha = np.exp(-t_dist * phi_temporal) 
        ax.scatter( y_obs, -s_obs, label="Observations", alpha=t_alpha, marker="x", color="green")
        ax.plot( np.exp(intensity_now),-s_porfile, label="Ture lambda", color="black")
        ax.plot( np.exp(mu_pred),-s_porfile, label="Predicted lambda", color="red")

        # locations that are possible next locations
        ax.scatter( np.zeros(len(s_next)),-s_next, label="Possible next locations", color="blue")

        # add the choosen next location
        ax.scatter(0, -s_now, label="Current location", color="black")
        ax.scatter(0,-next_location["s"], label="Choosen next location", color="red")
        
        ax.legend()
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Plancton intensity")
        ax.set_xlim([-4, 43])
        
        fig.savefig("figures/simulation/depth_profile_" + str(iter_i) + ".png", dpi=300)
        plt.close()

    # Print timing data
    t_alg = t_alg_end - t_alg_start
    print(f"Time for algorithm {t_alg:.2} seconds")

    if t_alg > max_planning_time:
        model.print_data_shape()
        model.down_sample_points()
        model.print_data_shape()

    if (iter_i % 100) == 0:
        print("Timing data")
        model.print_timing_data()
        #plot_timing(model.timing_data

print(iter_i)
# Making a video
"""
fileList = []
for i in range(iter_i-1):
    fileList.append("figures/simulation/depth_profile_" + str(i+1) + ".png",)

# Make a 1 minute video
choose_fps = round(iter_i / 60)
writer = imageio.get_writer("figures/simulation/depth_profile_video" + '.mp4', fps=choose_fps)

for im in fileList:
    writer.append_data(imageio.imread(im))
writer.close()
"""


convergence = model.convergence_data
print("Convergence data")
plt.scatter(convergence["n_points"], convergence["time"], alpha=0.5, )
plt.axhline(y=max_planning_time, color="red", linestyle="--")
plt.xlabel("Number of points")
plt.ylabel("Time seconds")
plt.title("Time used to fit the model")
plt.show()

n_iter = convergence["iterations"]
convergence_time = convergence["time"]


print(" average time per iteration", np.mean(convergence_time))
print("Average iterations per loop", np.mean(n_iter))

dat = pd.DataFrame({"n_iter": n_iter, "time": convergence_time})

boxplot = dat.boxplot(by="n_iter", column="time")
plt.xlabel("Number of iterations")
plt.ylabel("Time seconds")
plt.title("Time used to fit the model")
plt.show()


# Make a boxplot of the time used for each iteration
plt.boxplot(convergence_time, labels=n_iter)
plt.xlabel("Number of iterations")
plt.ylabel("Time seconds")
plt.title("Time used to fit the model")
plt.show()


print(" ====== Simulation done ======")