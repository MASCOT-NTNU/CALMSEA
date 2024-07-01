from help_func import * 
import time 

def run_simulation(model,
                   field,
                   path_planner, 
                   objective_function, 
                   evaluation, 
                   t_lim,
                   max_planning_time=0.5,
                   do_evaluation=True, print_while_running=False, plot_simulation=False):

    simulation_data = {}



    # Paramers for the simulation
    t_now = t_start
    diving_speed = 5 / 60 # m/s, 5 m/min 
    sampling_frequency = 1/10 # Hz , one sample per minute
    phi_temporal = 1 / 3600 # 1/s, the temporal decay of the observations
    s_lim = [0, 50] # The depth range in m

    # Add the first observation at the surface
    s_start_samples = np.array([0,1,2,3,3])
    t_start_samples = np.array([0,60,120,180, 181]) + t_start
    y_start = field.get_observation_Y(s_start_samples, t_start_samples)
    model.add_new_values(s_start_samples, t_start_samples, y_start)


    # simulate an adaptive sampling strategy
    iter_i = 0

    simulation_time_start = time.time()
    while t_now < t_end:
        iter_i += 1
        if print_while_running:
            print("====== i =======", iter_i)

        t_alg_start = time.time()

        s_now = model.get_s_now()
        t_now = model.get_t_now()

        # Get the possible next locations
        s_next, t_next = get_possible_next_locations(s_now, t_now, diving_speed, sampling_frequency, s_lim)

    
        # Get the prediction for the possible next locations
        design_data = []
        for i in range(len(s_next)):
            # Predict the intensity at the next location
            s_pred = np.repeat(s_next[i], 2)
            t_pred = np.array([t_next[i], t_next[i]+1])
            mu_pred, Sigma_pred = model.predict(s_pred, t_pred)
            loc_dict = {"s": s_next[i], "t": t_next[i], "x_pred": mu_pred, "Sigma_pred": Sigma_pred}
            design_data.append(loc_dict)

        # Get the next location
        next_location = objective_function.objective_function(model, design_data)

        # Get the next observation
        s_next = np.repeat(next_location["s"], 2)
        t_next = np.array([next_location["t"], next_location["t"]+1])
        y_next = field.get_observation_Y(s_next, t_next)

        # Add the new observation to the model
        model.add_new_values(s_next, t_next, y_next)

        # Evaluate the model
        if do_evaluation:
            evaluation.update_model(model)
            evaluation.evaluate()

            if print_while_running:
                evaluation.print_current_evaluation()


        # Plot the simulation
        if plot_simulation:
            # Plot the current model
            plt.figure()
            model.plot_model()
            plt.plot(s_next, t_next, "x", color="black")
            plt.show()

            # Plot the current field
            plt.figure()
            field.plot_field()
            plt.plot(s_next, t_next, "x", color="black")
            plt.show()

        # Print the time used
        t_alg_end = time.time()
        if print_while_running:
            print("Time used in algorithm", t_alg_end - t_alg_start)


        
        if t_alg_end - t_alg_start > max_planning_time:
            model.down_sample_points()

        

    if print_while_running:
        model.print_timing_data()
    
    print("Simulation done. Total time: ", round(time.time() - simulation_time_start,1) , "seconds")
    return model, evaluation

if __name__ == "__main__":
    # Import the model

    from Field import Field
    from Model import Model
    from ObjectiveFunction import ObjectiveFunction
    from Evaluation import Evaluation
    from PathPlanner import PathPlanner

    import os 

    experiment_id = 1
    # Save the simulation data
    store_path = "/Users/ajolaise/Documents/Simulation_data/PoissonModel"
    store_path = "./data/simulation_result/test_sim"

    # Create a folder for the simulation
    simulation_folder_path = store_path + "/experiment_" + str(experiment_id)

    # Create the folder
    if not os.path.exists(simulation_folder_path):
        print("Creating folder: ", simulation_folder_path)
        os.mkdir(simulation_folder_path)


    print_model_while_running = False
    print_simulation_while_running = False
    plot_simulation = False
    objective_function_name = "max_expected_improvement"

    max_planning_time = 0.5 # seconds


    # Create a field
    field = Field()

    # Generate the field
    t_lim = [0, 24*3600] # The time range in seconds eq 24 hours
    s_lim = [0, 50] # The depth range in m
    n_s = 50 # The number of points in the depth range
    n_t = 50 # The number of points in the time range
    field.generate_field_x(s_lim, t_lim, n_s, n_t)

    # Create a model
    model = Model(prior_mean_function=prior_function, print_while_running=print_model_while_running)
    objective_function = ObjectiveFunction(objective_function_name)
    evaluation = Evaluation(model, field)


    # Paramers for the simulation
    t_start = 0
    t_now = t_start
    t_end = (24 * 3600-100) -2000
    max_planning_time = 0.5 # seconds

    diving_speed = 5 / 60 # m/s, 5 m/min 
    sampling_frequency = 1/60 # Hz , one sample per minute

    phi_temporal = 1 / 3600 # 1/s, the temporal decay of the observations


    # Run the simulation
    model, evaluation = run_simulation(model, field, objective_function, evaluation, t_start, t_end,
                                        max_planning_time=max_planning_time,)
    
  

    # Save the model
    evaluation.save_evaluation(simulation_folder_path)

    
    # Create a folder for the simulation



