
import numpy as np
import os
import time 
import math 
import rospy
import datetime
import pickle
import pandas as pd

from help_func import *

from AUV import AUV
from WGS import WGS
from Field import Field
from Boundary import Boundary
from PathPlanner import PathPlanner
from Prior import Prior
from ObjectiveFunction import ObjectiveFunction
from Model import Model

from Agent import Agent




class AgentSilCam(Agent):

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

        model = Model(prior=self.prior,
                               phi_z=model_parameters["phi_z"],
                               phi_yx=model_parameters["phi_yx"],
                               phi_temporal=model_parameters["phi_temporal"],
                               sigma=model_parameters["sigma"],
                               print_while_running=True)

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
        objective_function = ObjectiveFunction(objective_function_parameters["objective_function_name"],
                                               print_while_running=objective_function_parameters["print_while_running"])
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
        print(time_now_str(), "[INFO] [AGENT] yo-yo completed", self.__counter_yoyo)
        print(time_now_str(), "[INFO] [AGENT] deapth seeker completed", self.__counter_line)
        if self.__planning_mode == "depth_seeker":
            print(time_now_str(), "[INFO] [AGENT] end goal:", self.__goal_wp)

        # Check if the current waypoint is inside the boundary
        if not self.operation_boundary.is_loc_legal(a):
            print(time_now_str(), "[ERROR] [AGENT] The current waypoint is outside the boundary")
            print(time_now_str(), "[INFO] [AGENT] current location: ", a)
            a_lat, a_lon = WGS.xy2latlon(a[0], a[1])
            print(time_now_str(), "[INFO] [AGENT] current location lat, lon: ", a_lat, a_lon)
            closest_legal_loc = self.operation_boundary.get_closest_legal_loc(a)
            closest_legal_loc = np.array([closest_legal_loc[0],closest_legal_loc[1],0])
            closest_legal_loc_lat, closest_legal_loc_lon = WGS.xy2latlon(closest_legal_loc[0], closest_legal_loc[1])
            print(time_now_str(), "[INFO] [AGENT] closest legal location: ", closest_legal_loc)
            print(time_now_str(), "[INFO] [AGENT] closest legal location lat, lon: ", closest_legal_loc_lat, closest_legal_loc_lon)
            dist = self.path_planner.get_distance(a, closest_legal_loc)
            print(time_now_str(), "[INFO] [AGENT] distance to closest legal location: ", round(dist,2) , "m")
            return closest_legal_loc, []
        
        if len(next_wp_list) > 0:
            # There are still waypoints in the list
            print(time_now_str(), "[INFO] [AGENT] still waypoints in the plan")
            next_wp = next_wp_list[0]
            dist_next_wp = self.path_planner.get_distance(a, next_wp)
            if dist_next_wp < 5:
                print(time_now_str(), "[INFO] [AGENT] next wp in list too close", dist_next_wp, "m")
                self.plan_next_waypoint(a, next_wp_list[1:])
            next_wp_list = next_wp_list[1:]
            return next_wp, next_wp_list
        
        if self.__planning_mode == "yoyo":
            return self.plan_next_waypoint_yoyo(a, next_wp_list)
        else:
            if self.__goal_wp is None:
                return self.plan_next_waypoint_depth_seeking_start(a)
            else:
                return self.plan_next_waypoint_depth_seeking(a, self.__goal_wp)
    
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
            print(time_now_str(), "[INFO] [AGENT] path evaluated", path["waypoints"])
            design_data = {}
            design_data["x_pred"], design_data["Sigma_pred"] = self.model.predict(path["S"], path["T"])
            design_data_paths.append(design_data)

        


        # Get the expected improvement
        best_design, design_id, of_values = self.objective_function.objective_function(self.model, design_data_paths)
        next_wp_list = paths[design_id]["waypoints"]
        next_wp = next_wp_list[0]
        
        # Update the counter
        self.__counter_yoyo += 1
        if self.__counter_yoyo >= self.agent_parameters["n_yoyos"]:
            self.__planning_mode = "depth_seeking"

        return next_wp, next_wp_list
    
    def plan_next_waypoint_depth_seeking_start(self, a):

        print(time_now_str(), "[ACTION] [MODEL] Planning the next waypoint depth seeking starting")

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
            design_data_paths.append(design_data)

        # Get the expected improvement
        best_design, design_id, of_values = self.objective_function.objective_function(self.model, design_data_paths)
        next_wp_list = paths[design_id]["waypoints"]
        goal = next_wp_list[-1]
        self.__goal_wp = goal
        dist_to_goal = self.path_planner.get_distance(a, self.__goal_wp)

        print(time_now_str(), "[INFO] [AGENT] goal waypoint set at", self.__goal_wp)
        print(time_now_str(), f"[INFO] [AGENT] distance to goal: {dist_to_goal:.2f} m" )

        return self.plan_next_waypoint_depth_seeking(np.array(a), np.array(self.__goal_wp))
    
    def plan_next_waypoint_depth_seeking(self, a, goal):
        """
        Plan the next waypoint
        """
        print(time_now_str(), "[ACTION] [MODEL] Planning the next waypoint depth_seeking")

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
            design_data_paths.append(design_data)
        
        

        # Get the expected improvement
        best_design, design_id, of_values = self.objective_function.objective_function(self.model, design_data_paths)
        next_wp = paths[design_id]["waypoints"][-1]
        
        # Update the counter
        if np.linalg.norm(next_wp - goal) < 20:
            self.__goal_wp = None
            self.__counter_line += 1


        return next_wp, []

    def read_silc_csv(self, file_path, threshold = 0.8):
        """
        Read a csv file with the SILC data
        """
        # Load the csv file
        try:
            df = pd.read_csv(file_path)
        except:
            print(time_now_str(), "[ERROR] [AGENT] Could not read the SILC data")
            return {"T": []}

        # Filter the enties that are none 
        df = df[df["export name"] != "not_exported"]

        # Get the time
        time_stamp = df["timestamp"].values

        # Transform the time to seconds
        try:
            time_stamp = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in time_stamp]
        except:
            print(time_now_str(), "[ERROR] [AGENT] Could not convert the time to seconds")
            print(time_stamp)
            return {"T": []}
        time_stamp = [x.timestamp() for x in time_stamp]

        probablity_columns = df.columns[9:16]
        copopod_data = df['probability_copepod'].values
        max_probability_df = df[probablity_columns]
    
        # for each row get the max probability column
        max_probability = max_probability_df.idxmax(axis=1)

        max_probability = max_probability.values

        columns_considered = ["probability_copepod"]

        return_data = {"T": [], "probability_copepod": []}
        for i, prob in enumerate(max_probability):
            if prob in columns_considered:
                if copopod_data[i] > threshold:
                    return_data["T"].append(time_stamp[i])
                    return_data["probability_copepod"].append(copopod_data[i])

        if len(return_data["T"]) == 0:
            print(time_now_str(), "[INFO] [AGENT] No probable copopods observed")
        else:
            print(time_now_str(), "[INFO] [AGENT] Found", len(return_data["T"]), "probable copopods")
        return return_data
    
    def read_process_silc_data(self,  file_path, threshold = 0.8):
        """
        Read a csv file with the SILC data
        """
        # Load the csv file
        try:
            df = pd.read_csv(file_path)
        except:
            print(time_now_str(), "[ERROR] [AGENT] Could not read the SILC data")
            return {"T": []}

        # Get the time
        time_stamp = df["timestamp"].values

        # Transform the time to seconds
        try:
            time_stamp = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in time_stamp]
        except:
            print(time_now_str(), "[ERROR] [AGENT] Could not convert the time to seconds")
            print(time_stamp)
            return {"T": []}
        time_stamp = [x.timestamp() for x in time_stamp]

        copopod_data = df['probability_copepod'].values

        return_data = {"T": [], "probability_copepod": []}
        for i, prob in enumerate(copopod_data):
            if prob > threshold:
                return_data["T"].append(time_stamp[i])
                return_data["probability_copepod"].append(copopod_data[i])

        if len(return_data["T"]) == 0:
            print(time_now_str(), "[INFO] [AGENT] No probable copopods observed")
        else:
            print(time_now_str(), "[INFO] [AGENT] Found", len(return_data["T"]), "probable copopods")
        return return_data


    def merge_data(self, current_data, silc_data):
        """
        merge the two dictionaries
        """

        t_curr = np.array(current_data["T"])
        t_silc = np.array(silc_data["T"])

        # print the time the two time
        if len(t_curr) > 0 and len(t_silc) > 0:
            print(time_now_str(), f"[INFO] [AGENT] current_data['T'][-1]: {t_curr[-1]} silc_data['T'][-1]: {t_silc[-1]}")

        if len(t_silc) == 0:
            print(time_now_str(), "[INFO] [AGENT] silc_data bas no data")
            if "copepod_count" in current_data.keys():
                diff_len = len(t_curr) - len(current_data["copepod_count"])
                temp = np.zeros(len(t_curr))
                temp[:len(current_data["copepod_count"])] = current_data["copepod_count"]
                current_data["copepod_count"] = temp
            else:
                current_data["copepod_count"] = np.zeros(len(t_curr))

            return current_data

        correction_t = 0
        t_silc = t_silc + correction_t

        copepod_count = np.zeros(len(t_curr))

        for i, t in enumerate(t_silc):
            if np.min(np.abs(t_curr - t)) < 2:
                index = np.argmin(np.abs(t_curr - t))
                copepod_count[index] += 1

        current_data["copepod_count"] = copepod_count
        return current_data


    def update_current_data(self, current_data):
        loc_auv = self.auv.get_vehicle_pos()                            # Get the location of the vehicle
        current_data["S"].append([loc_auv[0], loc_auv[1], loc_auv[2]])  # This is where we get the position data from the vehicle
        lat, lon = WGS.xy2latlon(loc_auv[0], loc_auv[1])                # This is where we get the lat lon data from the vehicle
        current_data["lat"].append(lat)
        current_data["lon"].append(lon)
        current_data["depth"].append(loc_auv[2])                        # This is where we get the depth data from the vehicle
        current_data["salinity"].append(self.auv.get_salinity())        # This is where we get the salinity data from the vehicle
        current_data["temperature"].append(self.auv.get_temperature())  # This is where we get the temperature data from the vehicle
        current_data["chlorophyll"].append(self.auv.get_chlorophyll())  # This is where we get the chlorophyll data from the vehicle
        current_data["T"].append(time.time())                           # This is where we get the time data from the vehicle

        #if time.time() - self.time_start < 10 * 60:
        t1 = time.time()
        silc_data = self.read_process_silc_data(self.agent_parameters["silc_processed_csv_path"], self.agent_parameters["silc_threshold"])
        current_data = self.merge_data(current_data, silc_data)
        t2 = time.time()

        print(time_now_str(), "sum(merged_data['copepod_count']):", sum(current_data["copepod_count"]))
        if len(current_data["copepod_count"]) > 5:
            # print the latest 5  values
            print(time_now_str(), "current_data['copepod_count'][-5:]", current_data["copepod_count"][-5:])
        print(time_now_str(), "Time to read csv ", round(t2-t1,2), "s")

    


    def add_data_to_model(self, current_data):
        """
        Add data to the model
        """
        t1 = time.time()
        silc_data = self.read_process_silc_data(self.agent_parameters["silc_processed_csv_path"], self.agent_parameters["silc_threshold"])
        current_data = self.merge_data(current_data, silc_data)
        t2 = time.time()

        print(time_now_str(), "sum(merged_data['copepod_count']):", sum(current_data["copepod_count"]))
        print(time_now_str(), "Time to read csv ", round(t2-t1,2), "s")

        S = np.array(current_data["S"])
        T = np.array(current_data["T"])
        Y = np.array(current_data["copepod_count"])
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
        "border_file": "/src/csv/mausund_mission_area.csv",
        "file_type": "latlon"
    }

    model_parameters = {
        "phi_temporal": 1 / 3600, # 1/s, the temporal decay of the observations
        "phi_yx": 1 / 1000, # 1/m, the spatial decay of the observations
        "phi_z": 1 / 10, # 1/m, the spatial decay of the observations
        "sigma": 0.1, # The noise of the observations
        "tau": 1, # The noise of the observations
        "Beta": [0.5, 2, 1 / 6**2]
        }

    
    path_planner_parameters = {
        "print_while_running": True,
        "attack_angle": 10,
        "max_speed": 1, # m/s
        "n_directions": 8,
        "n_depth_points": 9,
        "yoyo_step_length": 600,
        "step_length": 100,
        "max_depth": 70,
        "sampling_frequency": 1/10,
        }
    
    prior_parameters = {
        "beta_0": 0.7341, # The mean intensity
        "beta_1": 0.7341, # The amplitude of the intensity
        "beta_2": 1 / 6**2, # related to the width of the intensity
        "peak_depth_min": 25, # The minimum depth of the peak
        "peak_depth_max": 35 # The maximum depth of the peak
    }

    objective_function_parameters = {
        "objective_function_name": "max_expected_improvement",
        "print_while_running": True
    }
    
    agent_parameters = {
        "max_planning_time": 4, # s
        "speed": 1, # m/s
        "diving_speed": 2, # m/s
        "sampling_frequency": 1, # Hz
        "planning_mode": "yoyo",
        "n_yoyos": 2, 
        "num_steps": 200,
        "print_data_while_running": True,
        "n_messages_mission_complete": 10,
        "max_mission_time": 3600 * 1.5,
        "max_submerge_time": 15 * 60,
        "silc_csv_path": "src/silc_data/RAW-STATS.csv",
        "silc_processed_csv_path": "src/silc_data/RAW-STATS_processed.csv",
        "silc_threshold": 0.95
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

    experiment_id = "agent_silcam_mausund_" + id_now_str()
    agent = AgentSilCam(experiment_id=experiment_id,
                     boundary_parameters=boundary_parameters,
                     path_planner_parameters=path_planner_parameters,
                     prior_parameters=prior_parameters,
                     model_parameters=model_parameters,
                     objective_function_parameters=objective_function_parameters,
                     agent_parameters=agent_parameters)
    agent.run()








                    

