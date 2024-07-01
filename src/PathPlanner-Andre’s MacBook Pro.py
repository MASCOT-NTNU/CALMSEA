import numpy as np
import datetime
from help_func import *

from Boundary import Boundary

class PathPlanner:

    """
    This class is used to plan the path of the AUV
    it plants different paths depending on the mode

    it returns a list of paths that the AUV can follow

    the paths should be in the form of a dictionary
    path = {
        "waypoints": [wp1, wp2, wp3 ... wpn],   # The waypoints that the AUV should in the path
        "S": [s1, s2, s3 ... sn],               # The data points that the AUV could measure along the path
        "T": [t1, t2, t3 ... tn]                # The time points that the AUV could measure along the path
    }
    
    """
    def __init__(self, boundary: Boundary,
                 max_depth=90,
                 print_while_running=True,
                 max_speed=1.0,
                 attack_angle=13,
                 sampling_frequency=1/10, 
                 n_directions=10,
                 n_depth_points=9, 
                 yoyo_step_length=1000,
                 step_length=100):

        self.boundary = boundary


        # AUV parameters
        self.max_speed = max_speed # m/s
        self.attack_angle = attack_angle # degrees
        self.sample_frequency = sampling_frequency # Hz
        self.auv_depth_range = [0, max_depth] # m
        self.depth_seeking_depths = [2, max_depth] # m

        # Path planning parameters
        self.n_directions = n_directions
        self.n_depth_points = n_depth_points
        self.yoyo_step_length = yoyo_step_length
        self.step_length = step_length
        self.print_while_running = print_while_running

      


    @staticmethod
    def get_distance(start, end):
        """
        This function gets the distance between two points
        """
        return np.linalg.norm(end - start)

    @staticmethod
    def get_distance_yx(start, end):
        """
        This function gets the distance between two points in the y-x plane
        """
        return np.linalg.norm(end[:2] - start[:2])
    
    @staticmethod
    def rad2grad(rad):
        return rad * 180 / np.pi
    
    @staticmethod
    def grad2rad(grad):
        return grad * np.pi / 180
    

    def get_attack_angle(self, start, end):
        """
        This function calculates the attack angle between two points
        """
        attack_angle_rad = self.get_attack_angle_rad(start, end) 
        attack_angle = self.rad2grad(attack_angle_rad)
        return attack_angle
    
    @staticmethod
    def get_attack_angle_rad(start, end):
        """
        This function calculates the attack angle between two points
        """
        delta_depth = end[2] - start[2]
        L = np.linalg.norm(end[:2] - start[:2])
        attack_angle = np.arctan(np.abs(delta_depth) / L)
        return attack_angle
    
    
    
    def __is_path_legal(self, a, b):
        """
        This function checks if a path is legal
        """
        # Check if the path is within the field limits
        if self.boundary.is_path_legal(a, b) == False:
            return False
        
        # Check if the path is within the depth limits
        if b[2] < self.auv_depth_range[0] or b[2] > self.auv_depth_range[1]:
            return False
        
        # Check if the path is within the attack angle
        attack_angle = self.get_attack_angle(a, b)
        if attack_angle > self.attack_angle:
            return False
        
        return True
    
    def join_consecutive_paths(self, paths):
        """
        This function joins a list of paths
        """
        n_paths = len(paths)
        if n_paths == 0:
            return {"waypoints": [], "S": [], "T": []}
        
        if n_paths == 1:
            return paths[0]
        
        waypoints = paths[0]["waypoints"]
        S = paths[0]["S"]
        T = paths[0]["T"]
        for i in range(1,len(paths)):
            path = paths[i]
            waypoints += path["waypoints"]
            S = np.concatenate([S, path["S"]])
            T = np.concatenate([T, path["T"]])
        
        return {"waypoints": waypoints, "S": S, "T": T}
    

    def straight_line_path(self, start, end, t_start=0):
        """
        This function generates a straight line path between two points
        """

        attack_angle = self.get_attack_angle(start, end)
        if attack_angle > self.attack_angle + 0.001: 
            if self.print_while_running:
                print(time_now_str(), "[Warning] [PathPlanner] Attack angle too high for straight line path")
                print("start", start, "end", end)
                print("attack_angle", attack_angle)
        path = {"waypoints": [start, end], "S": []}
        self.add_data_points(path, t_start)
        return path
    

    def get_probable_path(self, start, end, t_start=0):
        """
        This function generates a probable path between two points
        The auv will not move in a straight line between the points 
        """
        ## NOT WORKING
        attack_angle = self.get_attack_angle(start, end)

        if attack_angle > self.attack_angle - 1:
            # The attack angle is close to straight line
            return self.straight_line_path(start, end, t_start)
        depth_end = end[2]
        depth_start = start[2]
        depth_delta = np.abs(depth_end - depth_start)

        dist_ac = depth_delta / np.tan(attack_angle)
        dist_cb = self.get_distance_yx(start, end) - dist_ac
        c = np.empty(3)
        c[2] = depth_end
        c[:2] = start[:2] + dist_ac * (end[:2] - start[:2]) / np.linalg.norm(end[:2] - start[:2])
        path_ac = self.straight_line_path(start, c, t_start)
        path_cb = self.straight_line_path(c, end, t_start + path_ac["T"][-1])
        path = {}
        print(path_ac) #REMOVE
        print(path_cb) #REMOVE
        path["waypoints"] = path_ac["waypoints"][0] + path_cb["waypoints"][-1]
        path["S"] = np.concatenate([path_ac["S"], path_cb["S"]])
        path["T"] = np.concatenate([path_ac["T"], path_cb["T"]])
        return path


    
    
    
    def move_towards_point(self, start, end, angle, target_depth):
        """
        This function moves the AUV towards a point with a certain angle and depth
        """
        delta_depth = target_depth - start[2]
        l = np.abs(delta_depth) / np.tan(angle)
        L = np.linalg.norm(end[:2] - start[:2])
        
        next_point_yx = start[:2] + l * (end[:2] - start[:2]) / L
        next_point = np.array([next_point_yx[0], next_point_yx[1], target_depth])
        return next_point
    

    def add_data_points(self,path, t_start=0):
        """
        This function adds the position and times of the possible data points
        """
        n_waypoints = len(path["waypoints"])
        data_points = []
        for i in range(n_waypoints-1):
            start = path["waypoints"][i]
            end = path["waypoints"][i+1]
            dist = np.linalg.norm(end - start)
            time = dist / self.max_speed
            n_points = int(np.floor(time * self.sample_frequency))
            data_points.append(np.linspace(start, end, n_points))
        path["S"] = np.concatenate(data_points)
        n_data_points = len(path["S"])

        # Add the time points
        path["T"] = np.linspace(t_start, t_start + n_data_points / self.sample_frequency, n_data_points)


    def get_possible_paths_z(self, start, end, path_depth_limits, t_start=0, min_depth_step=1):
        """
        Here the AUV is moving towards some target point, the target decies the direction in the y-x plane
        The AUV can can choose the depth that it keeps moving at

        a - start
        target - the point after the step length b = (b1, b2, b3 ...) these are the possible points
        c - target
        """

        paths = []

        # The number of depths we are aiming for
        n_depths = self.n_depth_points

        # Check if the target is less that the step_length away
        if np.linalg.norm(end - start) < self.step_length:
            # If it is we can just do a straight line path, this will give a single path
            # check if the target is within the attack angle
            if self.get_attack_angle(start, end) > self.attack_angle:
                if self.print_while_running:
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(now_str, "[WARNING] [PathPlanner] Attack angle too high in get_possible_paths_z")
            return [self.straight_line_path(start, end, t_start)]
        
        # Get the yx coordinates after the step length
        yx_target = start[:2] + self.step_length * (end[:2] - start[:2]) / np.linalg.norm(end[:2] - start[:2])
        dist_yx_b_c = np.linalg.norm(end[:2] - yx_target)

        # Get the depth limits
        target_depth_limits = self.get_depth_range_from_point(end, dist_yx_b_c)
        start_depth_limits = self.get_depth_range_from_point(start, self.step_length)
        all_depth_limits = [start_depth_limits, target_depth_limits, path_depth_limits, self.auv_depth_range, self.depth_seeking_depths]
        depth_limits_joined = self.join_depth_ranges(all_depth_limits)

        if depth_limits_joined[0] > depth_limits_joined[1]:
            # Ignore the depth limits regarding the path_depth_limits
            depth_limits_joined = self.join_depth_ranges([start_depth_limits, target_depth_limits, self.auv_depth_range])

        if depth_limits_joined[0] > depth_limits_joined[1] + 0.01:
            if self.print_while_running:
                print(time_now_str(), "[WARNING] [PathPlanner] error with the depth limits")
                print("depth_limits_joined", depth_limits_joined)
                print("path_depth_limits", path_depth_limits)
                print("target_depth_limits", target_depth_limits)
                print("start_depth_limits", start_depth_limits)
                print("auv_depth_range", self.auv_depth_range)
                print("start", start)
                print("target", end)

    
            return [self.straight_line_path(start, end, t_start)]
      

        if np.abs(depth_limits_joined[0] - depth_limits_joined[1]) < min_depth_step:
            if self.print_while_running:
                print(time_now_str(), "[WARNING] [PATH PLANNER] Depth range too narrow in get_possible_paths_z")
                print("depth_limits_joined", depth_limits_joined)
            return [self.straight_line_path(start, end, t_start)]

        # Finding the max number of depths that can be used
        max_points_in_range = int(np.floor((depth_limits_joined[1] - depth_limits_joined[0]) / min_depth_step) + 1)
        if max_points_in_range < n_depths:
            n_depths = max_points_in_range

        # Get the possible depths
        depths = np.linspace(depth_limits_joined[0], depth_limits_joined[1], n_depths)
        
        # Get the possible paths
        for i in range(n_depths):
            path = self.straight_line_path(start, np.array([yx_target[0], yx_target[1], depths[i]]), t_start)
            paths.append(path)

        if len(paths) == 1:
            if self.print_while_running:
                print(time_now_str(), "[INFO] [PathPlanner] Only one path to consider in get_possible_paths_z")
            return [self.straight_line_path(start, end, t_start) ]
        
        return paths
    
    
    def get_depth_range_from_point(self, point, step_length):
        """
        This function gets the depth range from a point
        From a point we can get to a maximum depth from the point and a minimum depth from the point
        This is done by moving using the attack angle and the auv depth range
        """
        max_depth_change = step_length * np.tan(self.attack_angle / 180 * np.pi)
        depth_range = [point[2] - max_depth_change, point[2] + max_depth_change]
        return depth_range
    
    def join_depth_ranges(self, depth_ranges):
        """
        This function joins a list of depth ranges
        """
        depth_range_joined = [0, 100] # The default depth range for the AUV is 0 to 100
        for depth_range in depth_ranges:
            if depth_range[0] > depth_range[1]:
                if self.print_while_running:
                    print(time_now_str(), "[WARNING] [PathPlanner] Error with the depth range")
                    print("depth_range", depth_range)
            
            depth_range_joined[0] = max(depth_range_joined[0], depth_range[0])
            depth_range_joined[1] = min(depth_range_joined[1], depth_range[1])
        depth_ranges = np.array(depth_ranges)
        joined_depth_range = [np.max(depth_ranges[:,0]), np.min(depth_ranges[:,1])]
        if joined_depth_range[0] > joined_depth_range[1]:
            if self.print_while_running:
                print(time_now_str(), "[WARNING] [PathPlanner] Error with the depth range")
                print("depth_ranges", depth_ranges)
                print("joined_depth_range", joined_depth_range)
                joined_depth_range = [min(joined_depth_range), min(joined_depth_range) + 1]
                if joined_depth_range[0] < 0:
                    joined_depth_range[0] = 0
                if joined_depth_range[1] < 0:
                    joined_depth_range[1] = 0
                print("fixed joined_depth_range", joined_depth_range)
        return joined_depth_range

    
    def yo_yo_path(self, start, end, depth_limits, t_start=0):
        """
        This function generates a yo-yo path between two points
        """
        depth_max = max(depth_limits)
        depth_min = min(depth_limits)
        a_depth = start[2]
        b_depth = end[2]
        a_yx = start[:2]
        b_yx = end[:2]
        L = np.linalg.norm(b_yx - a_yx)
        h = depth_max - depth_min
        alpha = self.attack_angle / 180 * np.pi  # convert to radians
        delta_depth = b_depth - a_depth

        # Yo-yo length
        l = 2 * h / np.tan(alpha)


        L_marked = L - np.abs(delta_depth) / np.tan(alpha)
        n = np.floor(L_marked / l)

        n_marked = (L - l * (np.abs(delta_depth) / h)) / l
        n = int(np.floor((L - l * (np.abs(delta_depth) / h)) / l))
        alpha_marked = np.arctan(2*n*h/L_marked)
        path = {"waypoints": [], "S": []}
        if l > L:
            # Check angle of attack
            attack_angle = np.arctan(np.abs(delta_depth) / L) * 180 / np.pi
            if attack_angle > self.attack_angle :
                if self.print_while_running:
                    print(time_now_str(), "[WARNING] [PATH PLANNER] Attack angle too high")
                    print("attack_angle", attack_angle, "self.attack_angle", self.attack_angle)
                return path

            # Now we need to do a straight line path if possible
            path = self.straight_line_path(start, end)
            return path
        
        # Move to a_marked. a_marked is the same depth as b_depth
        d_a_marked = np.abs(b_depth - a_depth) / np.tan(alpha)
        a_yx_marked = a_yx + d_a_marked * (b_yx - a_yx) / L
        a_depth_marked = b_depth
        a_marked = np.array([a_yx_marked[0], a_yx_marked[1], a_depth_marked])

        path = {"waypoints": [], "S": []}
        path["waypoints"].append(start)


        current_point = a_marked
        for i in range(int(n)*2):
            if delta_depth < 0:
                i = i + 1
            # We are diving 
            if i % 2 == 0:
                next_point = self.move_towards_point(current_point, end, alpha_marked, depth_max)
                path["waypoints"].append(next_point)
                current_point = next_point
            else:
                next_point = self.move_towards_point(current_point, end, alpha_marked, depth_min)
                path["waypoints"].append(next_point)
                current_point = next_point

        # Now we need to go straight to the end
        path["waypoints"].append(end)
        self.add_data_points(path, t_start=t_start)
        return path
    
    def suggest_directions(self, start, path_length=1000):
        """
        This function suggests n_directions for the AUV to follow
        the directions are in the y-x plane
        """
        possible_angles = np.linspace(0, 2*np.pi, self.n_directions)[0:-1]
        random_angles = possible_angles + np.random.uniform(0,np.pi / self.n_directions)

        # Getting the current point
        end_points = []
        for angle in random_angles:
            end_point = start[:2] + path_length * np.array([np.cos(angle), np.sin(angle)])
            end_point = np.array([end_point[0], end_point[1], start[2]]) # Add the depht dimension

            if self.__is_path_legal(start, end_point):
                end_points.append(end_point)
        
        return end_points
    
    def suggest_yoyo_paths(self,s_start, t_start):
        end_points = self.suggest_directions(s_start, self.yoyo_step_length)
        yoyo_paths = []
        t_b_s = 1 / self.sample_frequency
        for end_point in end_points:
            yoyo_path = self.yo_yo_path(s_start, end_point, self.auv_depth_range, t_start=t_start + t_b_s)
            yoyo_paths.append(yoyo_path)
        
        return yoyo_paths

    def __is_path_legal(self, a, b):
        """
        This function checks if a path is legal
        """
        if self.boundary.is_path_legal(a, b) == False:
            return False
        if b[2] < self.auv_depth_range[0] or b[2] > self.auv_depth_range[1]:
            return False
        attack_angle = self.get_attack_angle(a, b)
        if attack_angle > self.attack_angle:
            return False
        return True
        


    def get_random_path(self, s_start, t_start, distance=100, counter=0):
        random_attack_angle = np.random.uniform(-self.attack_angle, self.attack_angle)
        random_direction = np.random.uniform(0, 2*np.pi)
        end_point = s_start + distance * np.array([np.cos(random_direction), np.sin(random_direction), np.tan(random_attack_angle)])
        if self.__is_path_legal(s_start, end_point):
            path = self.straight_line_path(s_start, end_point, t_start)
            return path
        else:
            if counter > 100:
                if self.print_while_running:
                    print(time_now_str(), "[ERROR] [PathPlanner] Could not find a legal path")
                    print(counter, "tries")
                return None
            return self.get_random_path(s_start, t_start, distance, counter + 1)

    

    def suggest_next_paths(self, s_start, t_start, s_target=np.empty(3), mode="yoyo"):
        """
        This function plans the next waypoint for the AUV
        """
        
        if mode == "yoyo":
            yoyo_paths = self.suggest_yoyo_paths(s_start, t_start)
            return yoyo_paths
        if mode == "depth_seeking":
            path_depth_limits = [s_start[2] - 15, s_start[2] + 15]
            return self.get_possible_paths_z(s_start, s_target, path_depth_limits, t_start=t_start)

        print(time_now_str(), "[ERROR] [PathPlanner] Mode not recognized")
        return self.straight_line_path(s_start, s_target, t_start)
    
    def set_n_depth_points(self, n_depths, print_now=True):
        if self.print_while_running and print_now:
            print(time_now_str(), "[INFO] [PathPlanner] Setting n_depths_points to", n_depths)
        self.n_depth_points = n_depths


    def max_depth_at_time_t(self, time_remaining):
        """
        This function returns the maximum depth at time t
        """
        # The maximum depth is the depth that the AUV can reach in the time remaining

        attack_angle = self.attack_angle
        speed = self.max_speed
        max_dist = speed * time_remaining
        max_depth_change = max_dist * np.tan(attack_angle / 180 * np.pi)

        return max_depth_change





        
    
    def suggest_paths(self, s_start, t_start,time_remaining_submerged,
                        step_length = 100,
                        n_depths = 5,
                        n_directions=10):
        """
        This function suggests paths for the AUV
        """
        paths = []
        directions = self.suggest_directions(s_start, step_length)
        depth_limit_start = self.get_depth_range_from_point(s_start, step_length)
        depth_operation = self.depth_seeking_depths
        dep_limits_too_surface = [0, self.max_depth_at_time_t(time_remaining_submerged - step_length / self.max_speed)]
        joined_limits = self.join_depth_ranges([depth_limit_start, self.auv_depth_range, depth_operation, dep_limits_too_surface])

        min_depth_step = 1
        max_points_in_range = int(np.floor((joined_limits[1] - joined_limits[0]) / min_depth_step) + 1)
        n_depths = min(n_depths, max_points_in_range)
        depths = np.linspace(joined_limits[0], joined_limits[1], n_depths)
        for direction in directions:
            for depth in depths:
                end_point = np.array([direction[0], direction[1], depth])
                if self.__is_path_legal(s_start, end_point):
                    path = self.straight_line_path(s_start, end_point, t_start)
                    paths.append(path)
        return paths
    
    def suggest_multi_step_paths(self,
                                 s_start,
                                 t_start,
                                 time_remaining_submerged,
                                 n_steps=3,
                                 step_length=100,
                                 n_depths=5,
                                 n_directions=10):
        """
        This function suggests a multi step path
        """

        if n_steps <= 1:
            return self.suggest_paths(s_start, t_start, time_remaining_submerged, step_length, n_depths, n_directions)
        
        paths = []
        directions = self.suggest_directions(s_start, step_length)
        depth_limit_start = self.get_depth_range_from_point(s_start, step_length)
        depth_operation = self.depth_seeking_depths
        dep_limits_too_surface = [0, self.max_depth_at_time_t(time_remaining_submerged - step_length / self.max_speed)]
        joined_limits = self.join_depth_ranges([depth_limit_start, self.auv_depth_range, depth_operation, dep_limits_too_surface])

        min_depth_step = 2
        max_points_in_range = int(np.floor((joined_limits[1] - joined_limits[0]) / min_depth_step) + 1)
        n_depths = min(n_depths, max_points_in_range)
        depths = np.linspace(joined_limits[0], joined_limits[1], n_depths)
        for direction in directions:
            for depth in depths:
                end_point = np.array([direction[0], direction[1], depth])
                if self.__is_path_legal(s_start, end_point):
                    path = self.straight_line_path(s_start, end_point, t_start)
                    paths.append(path)

        multi_step_paths = []
        for i in range(len(paths)):
            path = paths[i]
            if n_steps == 1:
                return paths
            t_diff = path["T"][-1] - t_start
            next_paths = self.suggest_multi_step_paths(path["waypoints"][-1], path["T"][-1], time_remaining_submerged - t_diff,n_steps=n_steps - 1)
            for next_path in next_paths:
                multi_step_path = self.join_consecutive_paths([path, next_path])
                multi_step_paths.append(multi_step_path)

        return multi_step_paths

        
    def time_to_surface(self, depth):
        """
        This function returns the time to surface
        """
        attack_angle = self.attack_angle
        speed = self.max_speed
        max_dist = depth / np.tan(attack_angle / 180 * np.pi)
        time_to_surface = max_dist / speed
        return time_to_surface
    
    
    def get_optimum_yoyo_length(self):

        max_depth = self.auv_depth_range[1]
        attack_angle = self.attack_angle

        optimum_range = max_depth / np.tan(self.grad2rad(attack_angle)) * 2
        return optimum_range


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotting.PathPlanningPlotting import PathPlannerPlotting

    path_planner_plotting = PathPlannerPlotting()
    b = Boundary("/src/csv/simulation_border_xy.csv", file_type="xy")

    #########################################################
    ###### Test a random yo-yo between two points ###########
    #########################################################

    path_planner = PathPlanner(b)
    print("optimum yoyo length:", path_planner.get_optimum_yoyo_length())

    path_planner_plotting.plot_yoyo(path_planner)
    
   
    # Now do a lot of points to check if we get an error
    print("Testing 100 yo-yo paths")
    n = 100
    for i in range(n):
        start = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(0,90)])
        end = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(0,90)])
        yo_yo_delta = np.random.uniform(7, 19)
        yoyo_depth_limits = [end[2] + yo_yo_delta, end[2] - yo_yo_delta]
        if path_planner.get_attack_angle(start, end) < path_planner.attack_angle:
            path = path_planner.yo_yo_path(start, end, yoyo_depth_limits)



    print("#########################################################")
    print("###### Test get_possible_paths_z #######################")
    print("#########################################################")
        
    path_planner_plotting.plot_possible_paths_z(path_planner)

    # Now do a lot of points to check if we get an error
    n = 100
    for i in range(n):
    
        start = np.array([np.random.uniform(0,500), np.random.uniform(0,500), np.random.uniform(0,90)])
        end = np.array([np.random.uniform(1500,2000), np.random.uniform(1500,2000), np.random.uniform(0,90)])
        # Check the attack anglesy
        attack_angle = path_planner.get_attack_angle(start, end)
        if attack_angle < path_planner.attack_angle:
            path_delta = np.random.uniform(5, 50)
            path_depth_limits = [start[2] - path_delta, start[2] + path_delta]
            path_planner.set_n_depth_points(np.random.randint(3, 10), print_now=False)
            step_length = np.random.uniform(120, 240)
            path_planner.step_length = step_length
            paths = path_planner.get_possible_paths_z(start, end, path_depth_limits)
            
            


    #########################################################
    ##### Test many consecutive paths #######################
    #########################################################
    print("Testing many consecutive paths")
            
    path_planner_plotting.print_while_running = True
    path_planner_plotting.plot_consecutive_z_paths(path_planner)


    print("#########################################################")
    print("###### Test suggest_directions #######################")
    print("#########################################################")


    path_planner_plotting.plot_suggest_directions(path_planner)


    #########################################################
    ##### Test suggest_yoyo_paths #######################
    #########################################################

    path_planner_plotting.plot_suggest_yoyo_paths(path_planner)


    #########################################################
    ##### Test random mission #######################
    #########################################################

    path_planner_plotting.test_random_mission_path(path_planner)


    print("#########################################################")
    print("###### Test random paths #######################")
    print("#########################################################") 

    path_planner_plotting.plot_random_paths(path_planner)


    print("#########################################################")
    print("###### Test suggest_paths #######################")
    print("#########################################################")

    path_planner_plotting.plot_suggest_paths(path_planner)


    print("#########################################################")
    print("###### Test suggest_multi_step_paths #######################")
    print("#########################################################")

    path_planner = PathPlanner(b, n_directions=8, n_depth_points=5, sampling_frequency=1/20)
    path_planner_plotting.plot_suggest_paths_multi_step(path_planner)
