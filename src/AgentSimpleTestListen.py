
import numpy as np
import pandas as pd
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

from Agent import Agent



class SimpleAgentListen(Agent):

    def plan_next_waypoint(self, wp_now, wp_next_list=[]) -> np.ndarray:
        """
        Plan the next waypoint
        """
        print(time_now_str(), "[ACTION] [MODEL] Planning the next waypoint")
        return pre_planned_path(self.get_count()) , []
    

    def update_current_data(self, current_data):
        loc_auv = self.auv.get_vehicle_pos()                            # Get the location of the vehicle
        current_data["S"].append([loc_auv[0], loc_auv[1], loc_auv[2]])              # This is where we get the position data from the vehicle
        current_data["depth"].append(loc_auv[2])                        # This is where we get the depth data from the vehicle
        current_data["salinity"].append(self.auv.get_salinity())        # This is where we get the salinity data from the vehicle
        current_data["temperature"].append(self.auv.get_temperature())  # This is where we get the temperature data from the vehicle
        current_data["chlorophyll"].append(self.auv.get_chlorophyll())  # This is where we get the chlorophyll data from the vehicle
        current_data["T"].append(time.time()) 

        csv_data = self.read_csv(current_data, "random_data.csv")
        current_data["copopod"].append(csv_data)

    def read_csv(self, current_data, file_path):

        # Load the csv file
        try:
            df = pd.read_csv(file_path)
        except:
            print(print(time_now_str(), "[ERROR] [AGENT] Could not read the file: ", file_path))
            return None

        # Filter the enties that are none 

        # Get the time
        time_stamp = df["timestamp"].values

        # Transform the time to seconds
        time_stamp = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in time_stamp]
      

        copopod_data = df["copopod"].values

        # get correnting time 
        correcting_time = 0
        time_stamp_corrected = time_stamp - correcting_time
        
        # Get the current time
        time_stamp_current = current_data["T"]

        # Assign the data from df to a time_samp 
        current_comp = np.zeros(len(time_stamp_current))

        for i in range(len(time_stamp_corrected)):
            time_diff = time_stamp_current - time_stamp_corrected[i]
            index = np.argmin(np.abs(time_diff))
            current_comp[index] += copopod_data[i]

        return current_comp
        
    
if __name__ == "__main__":

    """
    This agent should just move in a square around the starting point
    """
    start_pos = np.empty(3)
    def pre_planned_path(counter):
        list_of_points = [np.array([0, 0,0]),
                          np.array([0, 200,0]),
                          np.array([200, 200,0]),
                          np.array([200, 0,0])]
        next_point_relative = list_of_points[int(counter) % len(list_of_points)]
        next_wp = next_point_relative + start_pos
        next_wp[2] = 1
        return next_wp
    

    experiment_id = "basic_agent_" +  id_now_str()
    agent = SimpleAgent(experiment_id=experiment_id)
    agent.set_number_of_steps(20)
    current_position = agent.auv.get_vehicle_pos()
    start_pos = current_position
    print(time_now_str(), "[INFO] [AGENT] current location: ", current_position)
    lat, lon = WGS.xy2latlon(current_position[0], current_position[1])
    print(time_now_str(), f"[INFO] [AGENT] current location in lat: {lat} lon: {lon}")
    agent.run()








                    

