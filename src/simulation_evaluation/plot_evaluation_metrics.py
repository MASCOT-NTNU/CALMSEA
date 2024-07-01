import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

print("Plotting evaluation metrics")

def sliding_average(x, window_size):
    return np.convolve(x, np.ones(window_size), 'valid') / window_size

simulation_id = "20231218_1706"
simulation_path = "./data/simulation_result/test_sim/simulation_" + simulation_id

# Get all files in the folder
file_names = os.listdir(simulation_path)


# Get the simulation parameter dicts
with open(simulation_path + "/simulation_parameter_dicts.pkl", "rb") as f:
    simulation_parameter_dicts = pickle.load(f)

parameters = {}
ignore_keys = ["experiment_id", "replicate_id"]
for parameter_dict in simulation_parameter_dicts:
    for key in parameter_dict.keys() - ignore_keys:
        if key not in parameters.keys():
            parameters[key] = [parameter_dict[key]]
        else:
            if parameter_dict[key] not in parameters[key]:
                parameters[key].append(parameter_dict[key])

print(parameters)




# Get the evaluation dicts
evaluation_dicts = []
for simulation_patameter_dict in simulation_parameter_dicts:
    exp_id = simulation_patameter_dict["experiment_id"]
    file_name = "evaluation_" + str(exp_id) + ".pkl"
    with open(simulation_path + "/" + file_name, "rb") as f:
        evaluation_dicts.append(pickle.load(f))

print(evaluation_dicts[1].keys())

eval_metrics = ["mse", "rmse", "dist_to_max", "dist_pred_max_to_max", "mse_lambda"]

# Plot the evaluation metrics

# Create a objective function color dict
objective_function_color_dict = {
    "max_expected_improvement": "blue",
    "max_variance": "red",
    "max_expected_intensity": "green",
    "random": "black",
    "yoyo_fast": "orange",
    "yoyo_slow": "purple"
}

"""
for evaluation_dict in evaluation_dicts:

    print(evaluation_dict.keys())
    for k in evaluation_dict.keys():
        print(k, len(evaluation_dict[k]))
"""

for eval_metric in eval_metrics:
    plt.figure()
    for i, evaluation_dict in enumerate(evaluation_dicts):
        objective_function_name = simulation_parameter_dicts[i]["objective_function_names"]
        plt.plot( evaluation_dict[eval_metric],lw = 1, color=objective_function_color_dict[objective_function_name])
    plt.legend([objective_function_name for objective_function_name in objective_function_color_dict.keys()])
    plt.title(eval_metric)
    plt.show()


# Average the evaluation metrics over the replicates
# Get the evaluation dicts



print(evaluation_dicts[0]["time_step"])

# Plot the evaluation metrics
for eval_metric in eval_metrics:
    sliding_average_window = 200
    plt.figure()
    plt.title("Sliding average " + eval_metric + " with window size " + str(sliding_average_window))
    for i, evaluation_dict in enumerate(evaluation_dicts):
        objective_function_name = simulation_parameter_dicts[i]["objective_function_names"]
        
        eval_avg = sliding_average(evaluation_dict[eval_metric], sliding_average_window)
        plt.plot(eval_avg, lw = 1, color=objective_function_color_dict[objective_function_name])

    for objective_function_name in objective_function_color_dict.keys():
        plt.axhline(y=0, color=objective_function_color_dict[objective_function_name], label=objective_function_name)
    plt.legend()
    plt.title(eval_metric)
    plt.show()

