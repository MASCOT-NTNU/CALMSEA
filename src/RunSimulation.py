import numpy as np
import time
import datetime


class RunSimulation:
    def __init__(self, 
                 model,
                 field,
                 objective_function,
                 path_planner,
                 evaluation = None,
                 depth_seek_limit = 10,
                 print_while_running = False, plot_simulation = False, max_planning_time = 60):
        
        self.model = model  # This is the statistical model
        self.field = field  # This is the simulated "real" field
        self.objective_function = objective_function # This is the objective function
        self.evaluation = evaluation # This is the evaluation function. And computes metrics for the simulation
        self.path_planner = path_planner # This finds and computes the possible paths in the field

        if self.evaluation is None and self.evaluation is True:
            raise ValueError("Evaluation function is not included")
        
        self.depth_seek_limit = depth_seek_limit
        
        self.max_planning_time = max_planning_time

        self.print_while_running = print_while_running
        self.plot_simulation = plot_simulation # This will increase the running time 

        # Simulation data
        self.simulation_data = {}

    def run_simulation(self):
        
        # Run a yo-yo survey
        if self.print_while_running:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now_str} [INFO] [SIMULATION] Running a yo-yo survey")
        self.yo_yo_survey()

        # Run a depth seeking survey
        if self.print_while_running:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now_str} [INFO] [SIMULATION] Running a depth seeking survey")
        

    def evaluate_paths(self, paths): 

        # Paths are a list of paths
        # each paths is structured like this
        # path = {"waypoints": [w1, w2, w3, ...,wM], "S": [s1, s2, s3, ..., sN], "T": [t1, t2, t3, ..., tN]}
        # where w1, w2, w3, ...,wM are the waypoints, this is guiding points for the AUV
        # s1, s2, s3, ..., sN and t1, t2, t3, ..., tN are the locations where the AUV will sample

        # The evaluation function will compute the metrics for the simulation
        design_data = []
        for path in paths:
            # Predict the intensity at the next location
            s_pred = path["S"]
            t_pred = path["T"]
            mu_pred, Sigma_pred = self.model.predict(s_pred, t_pred + 2)
            loc_dict = {"s": s_pred, "t": t_pred, "x_pred": mu_pred, "Sigma_pred": Sigma_pred}
            design_data.append(loc_dict)
        
        # here use the objective function to find the best path
        best_design, best_id = self.objective_function.objective_function(self.model, design_data)

        # Path evaluation data
        path_evaluation_data = {"best_design": best_design, "best_id": best_id, "design_data": design_data}

        return path_evaluation_data
    

    def move_agent(self, path):

        # The first step is to get the observations
        S_path, T_path = path["S"], path["T"]

        # add some noise to the locations
        S_path = S_path + np.random.normal(0, 0.1, S_path.shape)
        y_obs = self.field.get_observation_Y(S_path, T_path)

        # Add the new observations to the model
        self.model.add_new_values(S_path, T_path, y_obs)

        # now the agent has moved to the new location
        # when requesting s_now it will be the last point in the path



    def yo_yo_survey_step(self, yo_yo_depht_limits = [0, 50]):

        # Firsth we find the where the AUV is now
        s_now = self.model.get_s_now()
        t_now = self.model.get_t_now()

        # Find the possible directions
        end_points = self.path_planner.suggest_directions(s_now)

        # Find the possible yo-yo assosiated with the directions
        yo_yo_paths = []
        for end_point in end_points:
            t_b_s = 1 / self.path_planner.sample_frequency
            yo_yo_path = self.path_planner.yo_yo_path(s_now, end_point, yo_yo_depht_limits,t_start = t_now + t_b_s)
            yo_yo_paths.append(yo_yo_path)



        # Evaluate the paths
        path_evaluation_data = self.evaluate_paths(yo_yo_paths)
        path_evaluation_data["paths"] = yo_yo_paths
        return path_evaluation_data
    

    def depth_seeking_survey_step(self, end_point):
        # Firsth we find the where the AUV is now
        s_now = self.model.get_s_now()
        t_now = self.model.get_t_now()
        depth_seek_limit = self.depth_seek_limit

        z_now = s_now[2]
        depth_limits = [z_now - depth_seek_limit, z_now + depth_seek_limit]
        depth_limits = [max(depth_limits[0], 0), min(depth_limits[1], 50)]

        # Find the possible directions
        t_b_s = 1 / self.path_planner.sample_frequency
        paths = self.path_planner.get_possible_paths_z(s_now, end_point, depth_limits, t_start=t_now + t_b_s)

        # Evaluate the paths
        path_evaluation_data = self.evaluate_paths(paths)
        path_evaluation_data["paths"] = paths
        return path_evaluation_data
    
    def depth_seeking_survey_direction(self, end_point):

        """
        Here we move towards the end point in several steps, in each step we move in the direction of the end point
        but choose a different depth. The depth is chosen to maximize the expected intensity and end up in the correct depht
        at the end.
        """
            
        # Firsth we find the where the AUV is now
        s_now = self.model.get_s_now()
        t_now = self.model.get_t_now()

        step_number = 0

        survey_data = []

        while np.linalg.norm(s_now - end_point) > 20:

            if self.print_while_running:
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{now_str} [INFO] [SIMULATION] Step number: {step_number}")
            step_number += 1

            # Find the best path at this point in time
            path_evaluation_data = self.depth_seeking_survey_step(end_point)
            survey_data.append(path_evaluation_data)
            best_design = path_evaluation_data["best_design"]
            best_path = path_evaluation_data["paths"][path_evaluation_data["best_id"]]


            # TODO: add the option to plot the path
            # TODO: add down sampling here 

            # Move the agent to the next locationx
            self.move_agent(best_path)

            # Update the current location
            s_now = self.model.get_s_now()
            t_now = self.model.get_t_now()

        # Now the agent has arrived at thwe end location
            
        return survey_data

    def depth_seeking_survey(self):

        # Firsth we find the where the AUV is now
        s_now = self.model.get_s_now()
        t_now = self.model.get_t_now()

        survey_data = {"yo_yo_evaluation": None,
                        "depth_seeking_evaluation": None}


        # Find the possible directions
        end_points = self.path_planner.suggest_directions(s_now)

        # Get YOYO paths assosiated with the directions
        yo_yo_paths = []
        for end_point in end_points:
            yo_yo_path = self.path_planner.yo_yo_path(s_now, end_point,depth_limits =[0,50] ,  t_start = t_now)
            yo_yo_paths.append(yo_yo_path)

        if self.print_while_running:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now_str} [INFO] [SIMULATION] Evaluating {len(yo_yo_paths)} yo-yo paths")

        # Evaluate the paths
        path_evaluation_data = self.evaluate_paths(yo_yo_paths)
        path_evaluation_data["paths"] = yo_yo_paths

        survey_data["yo_yo_evaluation"] = path_evaluation_data

        # Find the end point with the best path
        best_end_point = path_evaluation_data["best_design"]["s"][-1]

        #TODO: add the option of first moving to the best depth

        if self.print_while_running:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now_str} [INFO] [SIMULATION] Moving to the point y={best_end_point[0]:.3} x={best_end_point[1]:.3} z={best_end_point[2]:.2}")

        # Move to the best end point
        depth_seeking_evaluation = self.depth_seeking_survey_direction(best_end_point)
        survey_data["depth_seeking_evaluation"] = depth_seeking_evaluation

        return survey_data

    def yo_yo_survey(self):
        # Firsth we find the where the AUV is now
        s_now = self.model.get_s_now()
        t_now = self.model.get_t_now()

        # Find the possible directions
        end_points = self.path_planner.suggest_directions(s_now)

        # Get YOYO paths assosiated with the directions
        yo_yo_paths = []
        for end_point in end_points:
            end_point_yo_yo = np.array([end_point[0], end_point[1], 0])
            yo_yo_path = self.path_planner.yo_yo_path(s_now, end_point_yo_yo,depth_limits =[0,50] ,  t_start = t_now)
            yo_yo_paths.append(yo_yo_path)

        if self.print_while_running:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now_str} [INFO] [SIMULATION] Evaluating {len(yo_yo_paths)} yo-yo paths")

        # Evaluate the paths
        path_evaluation_data = self.evaluate_paths(yo_yo_paths)
        path_evaluation_data["paths"] = yo_yo_paths

        # Store the evaluations 
        survey_data = path_evaluation_data

        # Find the end point with the best path
        best_end_point = path_evaluation_data["best_design"]["s"][-1]

        #TODO: add the option of first moving to the best depth

        if self.print_while_running:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now_str} [INFO] [SIMULATION] Moving to the point y={best_end_point[0]:.3} x={best_end_point[1]:.3} z={best_end_point[2]:.2}")

        
        # Move the Agent along the best yo-yo path
        self.move_agent(path_evaluation_data["paths"][path_evaluation_data["best_id"]])

        return survey_data

        




if __name__ == "__main__":

    from Field import Field
    from PathPlanner import PathPlanner
    from ObjectiveFunction import ObjectiveFunction
    from Model import Model

    import time
    from help_func import *
    from plotting.RunSimulationPlotting import RunSimulationPlotting

    

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
    print(f"Time to create the field: {t2 - t1:.2f} seconds")

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
    
    # Create the objective function
    print(" Creating the objective function ")
    objective_function = ObjectiveFunction(objective_function)

    # Create the path planner
    print(" Creating the path planner ")
    path_planner = PathPlanner(field_limits=s_lim, print_while_running=print_while_running,
                               max_speed=max_speed, 
                               attack_angle=attack_angle,
                               sampling_frequency=sampling_frequency,
                               yoyo_step_length=yoyo_step_length,
                               step_length=step_length)
    
    # Create the simulation
    print(" Creating the simulation ")
    simulation = RunSimulation(model, field, objective_function, path_planner, print_while_running=print_while_running, plot_simulation=plot_simulation, max_planning_time=max_planning_time)
    simulation_plotting = RunSimulationPlotting(simulation)



    ############################################
    #### Run some test for the simulation ######
    ############################################

    ###### One step with the depth seeking survey

    print("one step of the depth seeking survey")

    for i in range(7):

        model = Model(prior_mean_function=get_intensity_mu_x, 
                  phi_z=phi_z_model,
                  phi_yx=phi_yx_model,
                  phi_temporal=phi_t_model,
                  sigma=sigma_model,
                  print_while_running=print_while_running)
        simulation.model = model

        # Add the first observations at the surface
        m = 4
        sx_start = np.repeat(np.random.uniform(s_lim[0][0], s_lim[0][1]), m)
        sy_start = np.repeat(np.random.uniform(s_lim[1][0], s_lim[1][1]), m)
        sz_start = np.repeat(np.random.uniform(s_lim[2][0], s_lim[2][1]), m)
        s_start_samples = np.array([sx_start, sy_start, sz_start]).T
        t_start_samples = np.array([0,60,120,180])
        y_start = field.get_observation_Y(s_start_samples, t_start_samples)
        model.add_new_values(s_start_samples, t_start_samples, y_start)

        end_point = np.array([np.random.uniform(s_lim[0][0], s_lim[0][1]), np.random.uniform(s_lim[1][0], s_lim[1][1]), 25])
        path_evaluation_data = simulation.depth_seeking_survey_step(end_point)
        simulation_plotting.plot_depth_seek_one_step(path_evaluation_data, name=str(i))


    ###### Consecutive steps with the depth seeking survey
    # Initialize the model
    print("Consecutive steps of the depth seeking survey")
    model = Model(prior_mean_function=get_intensity_mu_x, 
                  phi_z=phi_z_model,
                  phi_yx=phi_yx_model,
                  phi_temporal=phi_t_model,
                  sigma=sigma_model,
                  print_while_running=print_while_running)
    simulation.model = model
    simulation_plotting.run_simulation = simulation

    # Add the first observations 
    m = 4
    sx_start = np.repeat(np.random.uniform(s_lim[0][0], s_lim[0][1]), m)
    sy_start = np.repeat(np.random.uniform(s_lim[1][0], s_lim[1][1]), m)
    sz_start = np.repeat(np.random.uniform(s_lim[2][0], s_lim[2][1]), m)
    s_start_samples = np.array([sx_start, sy_start, sz_start]).T
    t_start_samples = np.array([0,60,120,180])
    y_start = field.get_observation_Y(s_start_samples, t_start_samples)
    model.add_new_values(s_start_samples, t_start_samples, y_start)

    end_point = np.array([np.random.uniform(s_lim[0][0], s_lim[0][1]), np.random.uniform(s_lim[1][0], s_lim[1][1]), 25])
    d_start_end = np.linalg.norm(end_point[:2] - s_start_samples[0,:2])
    while d_start_end > 1000 or d_start_end < 500:
        end_point = np.array([np.random.uniform(s_lim[0][0], s_lim[0][1]), np.random.uniform(s_lim[1][0], s_lim[1][1]), 25])
        d_start_end = np.linalg.norm(end_point[:2] - s_start_samples[0,:2])
    simulation_plotting.plot_depth_seeker(end_point)
    simulation.model.print_timing_data()
    print(simulation.model.data_dict["batch"])


     ###### Consecutive steps with the depth seeking survey
    # Initialize the model
    print("Consecutive steps of the depth seeking survey")
    model = Model(prior_mean_function=get_intensity_mu_x, 
                  phi_z=phi_z_model,
                  phi_yx=phi_yx_model,
                  phi_temporal=phi_t_model,
                  sigma=sigma_model,
                  print_while_running=print_while_running)
    simulation.model = model
    simulation_plotting.run_simulation = simulation

    # Add the first observations 
    m = 4
    sx_start = np.repeat(np.random.uniform(s_lim[0][0], s_lim[0][1]), m)
    sy_start = np.repeat(np.random.uniform(s_lim[1][0], s_lim[1][1]), m)
    sz_start = np.repeat(np.random.uniform(s_lim[2][0], s_lim[2][1]), m)
    s_start_samples = np.array([sx_start, sy_start, sz_start]).T
    t_start_samples = np.array([0,60,120,180])
    y_start = field.get_observation_Y(s_start_samples, t_start_samples)
    model.add_new_values(s_start_samples, t_start_samples, y_start)

    end_point = np.array([np.random.uniform(s_lim[0][0], s_lim[0][1]), np.random.uniform(s_lim[1][0], s_lim[1][1]), 25])
    d_start_end = np.linalg.norm(end_point[:2] - s_start_samples[0,:2])
    while d_start_end > 1000 or d_start_end < 500:
        end_point = np.array([np.random.uniform(s_lim[0][0], s_lim[0][1]), np.random.uniform(s_lim[1][0], s_lim[1][1]), 25])
        d_start_end = np.linalg.norm(end_point[:2] - s_start_samples[0,:2])

    simulation.depth_seeking_survey_direction(end_point)
    s_now = simulation.model.get_s_now()
    print("Dist to end point", np.linalg.norm(s_now[:2] - end_point[:2]))

    ############################################
    ###### Depth seeking survey with the simulation
    ############################################

    print("Depth seeking survey with the simulation")
    model = Model(prior_mean_function=get_intensity_mu_x, 
                  phi_z=phi_z_model,
                  phi_yx=phi_yx_model,
                  phi_temporal=phi_t_model,
                  sigma=sigma_model,
                  print_while_running=print_while_running)
    simulation.model = model
    simulation_plotting.run_simulation = simulation

    m = 4
    sx_start = np.repeat(np.random.uniform(s_lim[0][0], s_lim[0][1]), m)
    sy_start = np.repeat(np.random.uniform(s_lim[1][0], s_lim[1][1]), m)
    sz_start = np.repeat(np.random.uniform(s_lim[2][0], s_lim[2][1]), m)
    s_start_samples = np.array([sx_start, sy_start, sz_start]).T
    t_start_samples = np.array([0,60,120,180])
    y_start = field.get_observation_Y(s_start_samples, t_start_samples)
    model.add_new_values(s_start_samples, t_start_samples, y_start)

    for i in range(20):
        simulation.depth_seeking_survey()
        if simulation.model.get_time_to_add_values()["time_list"][-1] > 1:  
            simulation.model.down_sample_points()
    timing_dict = simulation.model.timing_data
    plot_timing(timing_dict)
    simulation_plotting.plot_path_auv("depth_seeking")
    simulation_plotting.plot_path_auv_observation("depth_seeking")


    ############################################
    ###### YOYO survey with the simulation
    ############################################

    print("YOYO survey with the simulation")
    print("Depth seeking survey with the simulation")
    model = Model(prior_mean_function=get_intensity_mu_x, 
                  phi_z=phi_z_model,
                  phi_yx=phi_yx_model,
                  phi_temporal=phi_t_model,
                  sigma=sigma_model,
                  print_while_running=print_while_running)
    simulation.model = model
    simulation_plotting.run_simulation = simulation

    m = 4
    sx_start = np.repeat(np.random.uniform(s_lim[0][0], s_lim[0][1]), m)
    sy_start = np.repeat(np.random.uniform(s_lim[1][0], s_lim[1][1]), m)
    sz_start = np.repeat(0, m)
    s_start_samples = np.array([sx_start, sy_start, sz_start]).T
    t_start_samples = np.array([0,60,120,180])
    y_start = field.get_observation_Y(s_start_samples, t_start_samples)
    model.add_new_values(s_start_samples, t_start_samples, y_start)
    
    for i in range(20):
        simulation.yo_yo_survey()
        if simulation.model.get_time_to_add_values()["time_list"][-1] > 1:  
            simulation.model.down_sample_points()
            
    simulation_plotting.plot_path_auv("yo_yo")
    simulation_plotting.plot_path_auv_observation("yo_yo")
    timing_dict = simulation.model.timing_data
    simulation.model.print_timing_data()
    plot_timing(timing_dict)
                





    


        
        



