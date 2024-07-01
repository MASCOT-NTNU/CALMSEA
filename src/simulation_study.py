# Import the model

from Field import Field
from Model import Model
from ObjectiveFunction import ObjectiveFunction
from Evaluation import Evaluation
from run_simulation import run_simulation
from help_func import prior_function


import pickle
import itertools
import os 
import datetime
import numpy as np
import time

# Save the simulation data
store_path = "/Users/ajolaise/Documents/Simulation_data/PoissonModel"
store_path = "./data/simulation_result/test_sim"

# make a string on the form "YYYYMMDD_HHMMSS"
now = datetime.datetime.now()
now_str = now.strftime("%Y%m%d_%H%M")
print("now_str:", now_str)

# Set the seed for the random number generator
seed = int(now.strftime("%Y%m%d%H%M")) // 100
print("seed:", seed)
np.random.seed(seed)

print(now.strftime("%m/%d %H:%M:%S"))



# Create a folder for the simulation
simulation_folder_path = store_path + "/simulation_" + now_str

# Create the folder
if not os.path.exists(simulation_folder_path):
    print("Creating folder: ", simulation_folder_path)
    os.mkdir(simulation_folder_path)
else:
    print("Folder already exists. Exiting simulation")

    # Ask if the user wants to overwrite the folder
    user_input = input("Do you want to overwrite the folder? (y/n): ")
    if user_input == "y":
        print("Overwriting folder")
    else:
        print("Exiting simulation")
        exit()

print_model_while_running = False
print_simulation_while_running = False
plot_simulation = False


n_replicates = 20

simulation_parameter_dict = {
    "replicate_id": [i for i in range(n_replicates)],
    "objective_function_names": [ "yoyo_fast", "yoyo_slow", "random","max_expected_improvement", "max_variance", "max_expected_intensity"],
    "max_planning_time": [0.5]}

# Create a list of all the combinations of the parameters
simulation_parameter_list = list(itertools.product(*simulation_parameter_dict.values()))
simulation_parameter_dicts = []
experiment_ids = [i for i in range(len(simulation_parameter_list))]
for parameter_set in simulation_parameter_list:
    simulation_parameter_dicts.append(dict(zip(simulation_parameter_dict.keys(), parameter_set)))
# Add the experiment id to the parameter dict
for i in range(len(simulation_parameter_dicts)):
    simulation_parameter_dicts[i]["experiment_id"] = experiment_ids[i]


# Store the simulation dicts
with open(simulation_folder_path + "/simulation_parameter_dicts.pkl", "wb") as f:
    pickle.dump(simulation_parameter_dicts, f)






# Create a field
field = Field()

# Generate the field
t_lim = [0, 24*3600] # The time range in seconds eq 24 hours
s_lim = [0, 50] # The depth range in m
n_s = 50 # The number of points in the depth range
n_t = 100 # The number of points in the time range
print("Generating field")
field.generate_field_x(s_lim, t_lim, n_s, n_t)
field.save_field(simulation_folder_path, str(simulation_parameter_dicts[0]["replicate_id"]))

current_replicate_id = 0

for simulation_parameter_dict in simulation_parameter_dicts:
    print("=====================================")
    print("Simulation id: ", simulation_parameter_dict["experiment_id"], "of ", len(simulation_parameter_dicts))
    print("Running simulation with parameter set: ", simulation_parameter_dict)

    if simulation_parameter_dict["replicate_id"] != current_replicate_id:
        print("Generating new field")
        # Create a new field
        field = Field()
        field.generate_field_x(s_lim, t_lim, n_s, n_t)
        field.save_field(simulation_folder_path, str(simulation_parameter_dict["replicate_id"]))

        current_replicate_id = simulation_parameter_dict["replicate_id"]

    # load the parameters
    experiment_id = simulation_parameter_dict["experiment_id"]

    # Create a model
    model = Model(prior_mean_function=prior_function, print_while_running=print_model_while_running)
    objective_function = ObjectiveFunction(simulation_parameter_dict["objective_function_names"])
    evaluation = Evaluation(model, field)


    # Paramers for the simulation
    t_start = 0
    t_now = t_start
    t_end = (24 * 3600-100) -2000
    max_planning_time = simulation_parameter_dict["max_planning_time"] # seconds

    diving_speed = 5 / 60 # m/s, 5 m/min 
    sampling_frequency = 1/60 # Hz , one sample per minute

    phi_temporal = 1 / 3600 # 1/s, the temporal decay of the observations


    # Run the simulation
    t1 = time.time()
    model, evaluation = run_simulation(model, field, objective_function, evaluation, t_start, t_end,
                                        max_planning_time=max_planning_time,)
    t2 = time.time()
    print("Simulation time: ", round(t2 - t1,1))
    est_time_remaining = (t2 - t1) * (len(simulation_parameter_dicts) - experiment_id)/(60*60)
    print("Estimated time left: ", round(est_time_remaining, 2), " hours") 
    # This does not account for the time used to generate the field

    # Save the model
    evaluation.save_evaluation(simulation_folder_path, str(experiment_id))
    model.save(simulation_folder_path, str(experiment_id))

  