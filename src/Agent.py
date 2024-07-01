
import numpy as np
import os
import time 
import math 
import rospy
import datetime
import pickle


import sys
 
# adding src to the path to import the module
sys.path.insert(0, '/Users/ajolaise/Library/CloudStorage/OneDrive-NTNU/PhD/code/2023/PoissonModel3D/src/')

from help_func import *
from WGS import WGS
from Field import Field
from AUV import AUV






class Agent:

    def __init__(self,
                 experiment_id = "new",
                 boundary_parameters = {},
                 path_planner_parameters = {},
                 prior_parameters = {},
                 model_parameters = {},
                 objective_function_parameters = {},
                 agent_parameters = {"num_steps": 20,
                                     "print_data_while_running": True,
                                     "n_messages_mission_complete": 10,
                                     "speed": 1 
                                    }) -> None:
        """
        Setting up the agent 
        """
        print(time_now_str(), "[ACTION] [AGENT] Setting up the agent")
        
        # s0: setting up the parameters
        self.boudary_params = boundary_parameters
        self.path_planner_params = path_planner_parameters
        self.prior_parameters = prior_parameters
        self.model_parameters = model_parameters
        self.objective_function_parameters = objective_function_parameters

        # s1: set up the boundary
        self.operation_boundary = self.set_up_boundary(self.boudary_params)

        # s2: set up the path planner
        self.path_planner = self.set_up_path_planner(self.path_planner_params)

        # s3: set up the prior
        self.prior = self.set_up_prior(self.prior_parameters)
        
        # s3: set up the model
        self.model = self.set_up_model(self.model_parameters)

        # s4: set up Object function
        self.objective_function = self.set_up_objective_function(self.objective_function_parameters)

        # s5: set up the AUV
        self.auv = AUV()
        self.__loc_start = np.array(self.auv.get_vehicle_pos())
        print(time_now_str(), "[INFO] [AGENT] current location: ", self.__loc_start)
        lat, lon = WGS.xy2latlon(self.__loc_start[0], self.__loc_start[1])
        print(time_now_str(), f"[INFO] [AGENT] current location in lat: {lat} lon: {lon}")        


        # s6: storing variables
        self.agent_parameters = agent_parameters
        self.__counter = 0
        self.num_steps = agent_parameters["num_steps"]
        self.__print_data_while_running = agent_parameters["print_data_while_running"]
        self.__n_messages_mission_complete = agent_parameters["n_messages_mission_complete"]
        self.time_planning = []
        self.time_start = time.time()
        self.experiment_id = experiment_id
        self.parameters = {
            "experiment_id": self.experiment_id,
            "boudary_params": self.boudary_params,
            "path_planner_params": self.path_planner_params,
            "prior_parameters": self.prior_parameters,
            "model_parameters": self.model_parameters,
            "objective_function_parameters": self.objective_function_parameters,
            "agent_parameters": self.agent_parameters,
        }
        self.data = {}   # This is where the mission data is stored
        self.wp_next_list = []


         
        print(time_now_str(), "[INFO] [AGENT] Agent is set up and ready too run")


    def set_up_boundary(self, boundary_parameters=None):
        return None
    
    def set_up_path_planner(self, path_planner_params=None):
        return None
    
    def set_up_prior(self, prior_parameters=None):
        return None
    
    def set_up_model(self, model_parameters=None):
        return None
    
    def set_up_objective_function(self, objective_function_parameters=None):
        return None
    
    def plan_next_waypoint(self , current_wp, wp_next_list=[]) -> np.ndarray:
        pass

    def add_data_to_model(self, current_data):
        if self.model is None:
            return


    def run(self):

        
        # c1: start the operation from scratch.
        wp_start = self.__loc_start

        speed = self.auv.get_speed()
        max_submerged_time = self.auv.get_submerged_time()
        popup_time = self.auv.get_popup_time()
        phone = self.auv.get_phone_number()
        iridium = self.auv.get_iridium()

        # a1: move to current location
        self.set_next_waypoint(wp_start, speed)

        t_pop_last = time.time()
        update_time = rospy.get_time()


        # Setting up the data storage
        # This is the data we are getting from the vehicle
        current_data = { 
            "S": [],
            "depth": [],
            "salinity": [],
            "temperature": [],
            "chlorophyll": [],
            "T": []
        }


        # Plann the first waypoint
        # List of waypoints to go to
        current_wp = np.empty(3)
        wp_next = wp_start
        

        while not rospy.is_shutdown():
            if self.auv.init:

                t_now = time.time()

                print(time_now_str(), "counter: ", self.__counter, "\t vehicle state: ", self.auv.auv_handler.getState() ,end=" ")
                # s1: append data
                self.update_current_data(current_data)

                if self.__print_data_while_running:
                    self.print_current_data(current_data)

        


                # Check if the vehicle is waiting for a new waypoint
                if ((self.auv.auv_handler.getState() == "waiting") and
                        (rospy.get_time() - update_time) > 5.):
                    if t_now - t_pop_last >= max_submerged_time:
                        self.pop_up(popup_time, phone, iridium)
                        t_pop_last = time.time()
                    
                    # When we arrive at a waypoint we want the AUV to move as slow as possible
                    # Therfore we keep the old waypoint and set the speed to 0
                    #self.set_next_waypoint(wp_next, 1) 

                    # Timming the planning
                    t_plan_start = time.time()

                    # Checking if the points are legal
                    self.add_data_to_model(current_data)
                    
                    # Update the data storage
                    self.update_data(current_data)

                    # Reset the data storage
                    current_data = { 
                            "S": [],
                            "depth": [],
                            "salinity": [],
                            "temperature": [],
                            "chlorophyll": [],
                            "T": []
                        }

                    # Get the next waypoint
                    wp_next, self.wp_next_list = self.plan_next_waypoint(wp_next, self.wp_next_list)

                    # Set the next waypoint
                    self.set_next_waypoint(wp_next, speed)

                    # Update the time planning
                    self.time_planning.append(time.time() - t_plan_start)
                    print(time_now_str(), f"[INFO] [AGENT] time for planning:", round(self.time_planning[-1], 2))

                    # Update the counter 
                    self.print_counter()
                    self.__counter += 1

                    # Save data
                    self.save_data()

                    # downsample the data
                    self.downsampling()

                    # Sleep for 1 sec
                    time.sleep(1)
                
                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv.auv_handler.spin()

                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv.auv_handler.spin()

                if self.__counter == self.num_steps:
                    self.send_SMS_mission_complete(phone, iridium, popup_time)
                    break
            self.auv.rate.sleep()



    def downsampling(self):
        """
        Downsample the data
        """
        pass

    def get_curret_pos_xyz(self):
        loc_auv = self.auv.get_vehicle_pos()
        return np.array(loc_auv)

    def update_current_data(self, current_data):
        loc_auv = self.auv.get_vehicle_pos()                            # Get the location of the vehicle
        current_data["S"].append([loc_auv[0], loc_auv[1], loc_auv[2]])              # This is where we get the position data from the vehicle
        current_data["depth"].append(loc_auv[2])                        # This is where we get the depth data from the vehicle
        current_data["salinity"].append(self.auv.get_salinity())        # This is where we get the salinity data from the vehicle
        current_data["temperature"].append(self.auv.get_temperature())  # This is where we get the temperature data from the vehicle
        current_data["chlorophyll"].append(self.auv.get_chlorophyll())  # This is where we get the chlorophyll data from the vehicle
        current_data["T"].append(time.time())                           # This is where we get the time data from the vehicle

    def print_current_data(self, current_data):
        print(time_now_str(), "[INFO] [AGENT] Current data:")
        for key in current_data.keys():
            if key == "S":
                # Also print in Lat lon format
                lat, lon = WGS.xy2latlon(current_data[key][-1][0], current_data[key][-1][1])
                print(f"\t lat: {lat:.5f}, lon: {lon:.5f}", end=" \n")
            else:
                data_v = float(current_data[key][-1])
                print(f"\t {key}: {data_v:.3f}", end=" \n")


    def update_data(self, current_data):
        for key in current_data.keys():
            if key not in self.data.keys():
                self.data[key] = []
            self.data[key].extend(current_data[key])

    def set_next_waypoint(self, wp_next, speed) -> np.ndarray:
        # Going from x,y to lat,lon in the WGS84 system
        lat, lon, = WGS.xy2latlon(wp_next[0], wp_next[1])

        # Set the waypoint to the vehicle 
        print(time_now_str(), "[ACTION] [AGENT] Setting waypoint]")
        print(time_now_str(), f"[INFO] [AGENT] next waypoint x: {wp_next[0]}, y: {wp_next[1]}, depth {wp_next[2]}")
        print(time_now_str(), f"[INFO] [AGENT] next waypoint lat: {lat}, lon: {lon}, depth {wp_next[2]}")
        self.auv.auv_handler.setWaypoint(math.radians(lat), math.radians(lon), wp_next[2], speed=speed)
        distance_to_waypoint = np.linalg.norm(wp_next  - self.get_curret_pos_xyz())
        print(time_now_str(), f"[INFO] [AGENT] distance to next waypoint {distance_to_waypoint:.2f} m")
        time_to_waypoint = distance_to_waypoint / speed / 60 # in minutes
        print(time_now_str(), f"[INFO] [AGENT] time to next waypoint {time_to_waypoint:.2f} min")
        print(time_now_str(), "[ACTION] Waypoint sent to auv_handler")


    def send_SMS_mission_complete(self, phone, iridium, popup_time):
        print(time_now_str(), "[INFO] Mission complete!")

        for i in range(self.__n_messages_mission_complete):
            self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                        phone_number=phone, iridium_dest=iridium)
        print(time_now_str(), "[INFO] [AGENT] pop-up message sent")
        rospy.signal_shutdown("done")

    def pop_up(self, popup_time, phone, iridium):
        print("[ACTION] [AGENT] Popping up]")
        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                    phone_number=phone, iridium_dest=iridium)
        print("[ACTION] [AGENT] pop-up message sent")

    def get_count(self):
        return self.__counter

    
    def save_data(self):
        print(time_now_str(), "[INFO] [AGENT] Saving data")

        # Create the folder if it does not exist
        if not os.path.exists(f"src/mission_data/{self.experiment_id}"):
            os.makedirs(f"src/mission_data/{self.experiment_id}")
        with open(f"src/mission_data/{self.experiment_id}/data_{self.__counter}.pkl", "wb") as f:
            pickle.dump(self.data, f)
        with open(f"src/mission_data/{self.experiment_id}/parameters_{self.__counter}.pkl", "wb") as f:
            pickle.dump(self.parameters, f)
        if self.model is not None:
            self.model.save(f"src/mission_data/{self.experiment_id}/", str(self.__counter))

    def print_counter(self):
        print("-----------------------------------------------------")
        print("#################   Counter", self.__counter + 1, "   #################")
        print("-----------------------------------------------------")

    def set_number_of_steps(self, num_steps):
        self.num_steps = num_steps
    
if __name__ == "__main__":

    experiment_id = "pre_mission_test_" + id_now_str()
    Agent = Agent(experiment_id=experiment_id)
    Agent.run()








                    

