
import numpy as np
import os
import time 
import math 
import rospy
import datetime
import pickle

from help_func import *

from AUV import AUV
from WGS import WGS
from Field import Field
from Boundary import Boundary
from PathPlanner import PathPlanner
from Prior import Prior
from ObjectiveFunction import ObjectiveFunction

from ModelLogNormal import ModelLogNormal

from Agent import Agent




class AgentCla(Agent):

    __counter_yoyo = 0
    __counter_line = 0
    __planning_mode = "yoyo"
    __goal_wp = None


    def set_up_prior(self, prior_parameters=None):
        """
        Set up the prior
        """
        print(time_now_str(), "[ACTION] [AGENT] setting up the prior")
        print(time_now_str(), "[INFO] [AGENT] parameters:", prior_parameters)
        prior = Prior(parameters=prior_parameters)
        print(time_now_str(), "[INFO] [AGENT] Prior is set up")
        return prior


    def set_up_model(self, model_parameters):
        """
        Set up the model
        """

        print(time_now_str(), "[ACTION] [AGENT] setting up the model")
        print(time_now_str(), "[INFO] [AGENT] parameters:", model_parameters)

        model = ModelLogNormal(prior=self.prior,
                               phi_z=model_parameters["phi_z"],
                               phi_yx=model_parameters["phi_yx"],
                               phi_temporal=model_parameters["phi_temporal"],
                               sigma=model_parameters["sigma"],
                               tau=model_parameters["tau"],
                               print_while_running=True)
        
        print(time_now_str(), "[INFO] [AGENT] model set up now")
        return model
    
    def set_up_boundary(self, boundary_parameters):
        """
        Set up the boundary
        """

        print(time_now_str(), "[ACTION] [AGENT] Setting up the boundary")
        print(time_now_str(), "[INFO] [AGENT] parameters:", boundary_parameters)
        boundary = Boundary(border_file=boundary_parameters["border_file"],
                            file_type=boundary_parameters["file_type"])
        print(time_now_str(), "[INFO] [AGENT] Boundary is now set up") 
        boundary.print_border_xy()
        boundary.print_border_latlon()       
        return boundary
    
    def set_up_path_planner(self, path_planner_params):
        """
        Set up the path planner
        """

        print(time_now_str(), "[ACTION] [AGENT] setting up the path planner")
        print(time_now_str(), "[INFO] [AGENT] parameters:", path_planner_parameters)
        path_planner = PathPlanner(boundary=self.operation_boundary,
                                    attack_angle=path_planner_params["attack_angle"],
                                    max_speed=path_planner_params["max_speed"],
                                    n_directions=path_planner_params["n_directions"],
                                    n_depth_points=path_planner_params["n_depth_points"],
                                    yoyo_step_length=path_planner_params["yoyo_step_length"],
                                    step_length=path_planner_params["step_length"],
                                    max_depth=path_planner_params["max_depth"],
                                    print_while_running=path_planner_params["print_while_running"])
        print(time_now_str(), "[INFO] [AGENT] Path planner set up")
        return path_planner
    
    def set_up_objective_function(self, objective_function_parameters):
        """
        Set up the objective function
        """
        print(time_now_str(), "[ACTION] [INFO] Setting up the objective function")
        print(time_now_str(), "[INFO] [AGENT] parameters:", objective_function_parameters)
        objective_function = ObjectiveFunction(objective_function_parameters["objective_function_name"])
        print(time_now_str(), "[INFO] [AGENT] Objective function set up")
        return objective_function
    
    def plan_next_waypoint(self, a, next_wp_list=[]) -> np.ndarray:
        """
        Plan the next waypoint
        """
        print(time_now_str(), "[ACTION] [AGENT] Planning the next waypoint")
        print(time_now_str(), "[INFO] [AGENT] current location: ", a)
        print(time_now_str(), "[INFO] [AGENT] next_wp_list: ", next_wp_list)
        print(time_now_str(), "[INFO] [AGENT] planning mode: ", self.__planning_mode)

        # Check if the current waypoint is inside the boundary
        if not self.operation_boundary.is_loc_legal(a):
            print(time_now_str(), "[ERROR] [AGENT] The current waypoint is outside the boundary")
            print(time_now_str(), "[INFO] [AGENT] current location: ", a)
            a_lat, a_lon = WGS.xy_to_latlon(a[0], a[1])
            print(time_now_str(), "[INFO] [AGENT] current location lat, lon: ", a_lat, a_lon)
            closest_legal_loc = self.operation_boundary.get_closest_legal_loc(a)
            closest_legal_loc = np.array([closest_legal_loc[0],closest_legal_loc[1],0])
            closest_legal_loc_lat, closest_legal_loc_lon = WGS.xy_to_latlon(closest_legal_loc[0], closest_legal_loc[1])
            print(time_now_str(), "[INFO] [AGENT] closest legal location: ", closest_legal_loc)
            print(time_now_str(), "[INFO] [AGENT] closest legal location lat, lon: ", closest_legal_loc_lat, closest_legal_loc_lon)
            dist = self.path_planner.get_distance(a, closest_legal_loc)
            print(time_now_str(), "[INFO] [AGENT] distance to closest legal location: ", round(dist,2) , "m")
            return closest_legal_loc, []
        
        if len(next_wp_list) > 0:
            # There are still waypoints in the list
            next_wp = next_wp_list[0]
            next_wp_list = next_wp_list[1:]
            return next_wp, next_wp_list
        
        if self.__planning_mode == "yoyo":
            return self.plan_next_waypoint_yoyo(a, next_wp_list)
        else:
            if self.__goal_wp is None:
                return self.plan_next_waypoint_depth_seeking_start(a)
            else:
                return self.plan_next_waypoint_depth_seeking(a, self.__goal)
    
    def plan_next_waypoint_yoyo(self, a, next_wp_list) -> np.ndarray:
        """
        Plan the next waypoint
        """
        print(time_now_str(), "[ACTION] [AGENT] Planning the next waypoint")

        t_start = self.model.get_t_now()
        paths = self.path_planner.suggest_next_paths(a, t_start, mode="yoyo")
        if len(paths) == 0:
            print(time_now_str(), "[ERROR] [AGENT] No paths found")
            return None, None
        if len(paths) == 1:
            print(time_now_str(), "[ACTION] [AGENT] Only one path found")
            next_wp_list = paths[0]["waypoints"]

        next_wp = paths[0]
        design_data_paths = []
        for path in paths:
            design_data = {}
            design_data["x_pred"], design_data["Sigma_pred"] = self.model.predict(path["S"], path["T"])

        


        # Get the expected improvement
        best_design, design_id = self.objective_function.objective_function(self.model, design_data_paths)
        next_wp_list = paths[design_id]["waypoints"]
        next_wp = next_wp_list[0]
        
        # Update the counter
        self.__counter_yoyo += 1

        return next_wp, next_wp_list
    
    def plan_next_waypoint_depth_seeking_start(self, a):

        print(time_now_str(), "[ACTION] [MODEL] Planning the next waypoint")

        t_start = self.model.get_t_now()
        paths = self.path_planner.suggest_next_paths(a, t_start, mode="yoyo")
        if len(paths) == 0:
            print(time_now_str(), "[ERROR] [AGENT] No paths found")
            return None, None
        if len(paths) == 1:
            print(time_now_str(), "[ACTION] [AGENT] Only one path found")
            next_wp_list = paths[0]["waypoints"]

        design_data_paths = []
        for path in paths:
            design_data = {}
            design_data["x_pred"], design_data["Sigma_pred"] = self.model.predict(path["S"], path["T"])

        # Get the expected improvement
        best_design, design_id = self.objective_function.objective_function(self.model, design_data_paths)
        next_wp_list = paths[design_id]["waypoints"]
        goal = next_wp_list[-1]
        
        return self.plan_next_waypoint(self, a, goal)
    
    def plan_next_waypoint_depth_seeking(self,a, goal):
        """
        Plan the next waypoint
        """
        print(time_now_str(), "[ACTION] [MODEL] Planning the next waypoint")

        t_start = self.model.get_t_now()
        paths = self.path_planner.suggest_next_paths(a, t_start,s_target = goal,  mode="depth_seeking")
        if len(paths) == 0:
            print(time_now_str(), "[ERROR] [AGENT] No paths found")
            return None, None
        if len(paths) == 1:
            print(time_now_str(), "[ACTION] [AGENT] Only one path found")
            next_wp_list = paths[0]["waypoints"]

        print(time_now_str(), f"[INFO] [AGENT] Evaluating {len(paths)} different paths")
        print(time_now_str(), f"[INFO] [AGENT] paths too evaluat:")

        next_wp = paths[0]
        design_data_paths = []
        for path in paths:
            print("\t ", path["waypoints"])
            design_data = {}
            design_data["x_pred"], design_data["Sigma_pred"] = self.model.predict(path["S"], path["T"])
        
        

        # Get the expected improvement
        best_design, design_id = self.objective_function.objective_function(self.model, design_data_paths)
        next_wp = paths[design_id]["waypoints"][-1]
        
        # Update the counter
        if np.linalg.norm(next_wp - goal) < 20:
            self.__goal_wp = None
            self.__counter_line += 1


        return next_wp, next_wp_list


    def add_data_to_model(self, current_data):
        """
        Add data to the model
        """
        S = np.array(current_data["S"])
        T = np.array(current_data["T"])
        Y = np.array(current_data["chlorophyll"])
        print(time_now_str(), "[ACTION] [AGENT] Adding data to the model")
        self.model.add_new_values(S, T, Y)
        print(time_now_str(), "[INFO] [AGENT] Data added to the model")


    def downsampling(self):
        """
        Downsample the data
        """
        if self.time_planning[-1] > self.agent_parameters["max_planning_time"]:
            print(time_now_str(), "[ACTION] [AGENT] Downsampling the data")
            self.model.down_sample_points()
            print(time_now_str(), "[INFO] [AGENT] Data downsampled")





        
    

        
    
        
        
    
if __name__ == "__main__":

    """
    Setting up the parameters for the agent
    """

   
    
    boundary_parameters = {
        "border_file": "/src/csv/simulation_border_yx.csv",
        "file_type": "yx"}
    boundary_parameters = {
        "border_file": "/src/csv/hitl_xy.csv",
        "file_type": "xy"
    }

    model_parameters = {
        "phi_temporal": 1 / 3600, # 1/s, the temporal decay of the observations
        "phi_yx": 1 / 1000, # 1/m, the spatial decay of the observations
        "phi_z": 1 / 10, # 1/m, the spatial decay of the observations
        "sigma": 0.1, # The noise of the observations
        "tau": 0.1, # The noise of the observations
        "Beta": [0.5, 2, 1 / 6**2]
        }

    
    path_planner_parameters = {
        "print_while_running": True,
        "attack_angle": 10,
        "max_speed": 1, # m/s
        "n_directions": 8,
        "n_depth_points": 9,
        "yoyo_step_length": 500,
        "step_length": 100,
        "max_depth": 90,
        }
    
    prior_parameters = {
        "beta_0": 0.5, # The mean intensity
        "beta_1": 2, # The amplitude of the intensity
        "beta_2": 1 / 6**2, # related to the width of the intensity
        "peak_depth_min": 10, # The minimum depth of the peak
        "peak_depth_max": 60 # The maximum depth of the peak
    }

    objective_function_parameters = {
        "objective_function_name": "max_expected_improvement"
    }
    
    agent_parameters = {
        "max_planning_time": 2, # s
        "speed": 1, # m/s
        "sampling_frequency": 1, # Hz
        "planning_mode": "yoyo",
        "num_steps": 100,
        "print_data_while_running": True,
        "n_messages_mission_complete": 10
    }    

    ### Test create box latlon border file
    auv = AUV()
    current_loc = auv.get_vehicle_pos()
    print(time_now_str(), f"Center of border file (x,y,z) {current_loc}")
    current_loc_lat, current_loc_lon = WGS.xy2latlon(current_loc[0], current_loc[1])
    print(time_now_str(), f"Center of border file lat {current_loc_lat}, lon {current_loc_lon}")
    size = 2000 # meters
    file_name = "hitl"
    Boundary.create_box_xy_border_file_from_loc(current_loc, size, file_name, "xy")
    auv = None

    experiment_id = "agent_cla_" + id_now_str()
    agent = AgentCla(experiment_id=experiment_id,
                     boundary_parameters=boundary_parameters,
                     path_planner_parameters=path_planner_parameters,
                     prior_parameters=prior_parameters,
                     model_parameters=model_parameters,
                     objective_function_parameters=objective_function_parameters,
                     agent_parameters=agent_parameters)
    agent.run()








                    

