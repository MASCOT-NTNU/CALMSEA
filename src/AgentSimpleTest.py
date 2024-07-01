
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

from Agent import Agent



class SimpleAgent(Agent):

    def plan_next_waypoint(self, wp_now, wp_next_list=[]) -> np.ndarray:
        """
        Plan the next waypoint
        """
        print(time_now_str(), "[ACTION] [MODEL] Planning the next waypoint")
        return pre_planned_path(self.get_count()) , []
        
    
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








                    

