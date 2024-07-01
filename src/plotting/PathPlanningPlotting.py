
import numpy as np
import matplotlib.pyplot as plt

class PathPlannerPlotting:
    def __init__(self, print_while_running=False):
        self.print_while_running = print_while_running

    def plot_yoyo(self, path_planner):
        start = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(0,50)])
        end = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(0,50)])
        yoyo_depth_limits = [end[2] + 10, end[2] - 10]
        path = path_planner.yo_yo_path(start, end, yoyo_depth_limits)

        path_list = np.array(path["waypoints"])
        d_xy_list = np.linalg.norm(path_list[0,:2] - path_list[:, :2], axis=1)
        z_list = path_list[:, 2]
        data_points = path["S"]
        d_xy_data = np.linalg.norm(data_points[0,:2] - data_points[:, :2], axis=1)
        z_data = data_points[:, 2]
        plt.plot(d_xy_list, z_list, label="Waypoints")
        plt.plot(d_xy_data, z_data, 'o', label="S")
        # Plot two horizontal lines
        plt.axhline(y=min(yoyo_depth_limits), color='k', linestyle='--', label="Depth limits")
        plt.axhline(y=max(yoyo_depth_limits), color='k', linestyle='--')
        plt.plot(0, start[2], 'o', label="Start")
        plt.plot(d_xy_list[-1], end[2], 'o', label="End")
        plt.xlabel("Distance [m]")
        plt.ylabel("Depth [m]")
        plt.gca().invert_yaxis()
        plt.title("Yo-yo path between two points")
        plt.legend()
        plt.savefig("figures/tests/PathPlanner/yoyo_path.png")
        plt.close()

    def plot_possible_paths_z(self, path_planner):
        start = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(0,50)])
        end = start + np.random.uniform(-200, 200, 3)
        end[2] = np.random.uniform(0, 50)
        #end = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(0,50)])

        d_start_end = np.linalg.norm(end[:2] - start[:2])
        step_length = 100
        path_planner.step_length = 100
        d_b_c = d_start_end - step_length
        max_depth_change = d_b_c * np.tan(path_planner.attack_angle * np.pi / 180)
        max_depth_change_from_start = step_length * np.tan(path_planner.attack_angle * np.pi / 180)
        n_depths = 9 # Should aim for odd 
        path_planner.n_depths = n_depths

        path_depth_limits = [start[2] - 30, start[2] + 10]
        paths = path_planner.get_possible_paths_z(start, end, path_depth_limits)
        plt.plot(0, start[2], 'o', label="Start")
        plt.plot(d_start_end, end[2], 'o', label="Target")

        plt.axhline(y=path_depth_limits[0], color='k', linestyle='--', label="Target depth")
        plt.axhline(y=path_depth_limits[1], color='k', linestyle='--')

        plt.plot([0, step_length],[start[2], start[2] + max_depth_change_from_start], linestyle="--", c="red", label="Start depth limits")
        plt.plot([0, step_length],[start[2], start[2] - max_depth_change_from_start], linestyle="--", c="red")

        plt.axhline(y=0, color='g', linestyle='--', label="Field depth")
        plt.axhline(y=50, color='g', linestyle='--')

        plt.plot([d_start_end, step_length], [end[2] ,end[2] + max_depth_change],"--", c="red", label="Target depth limits")
        plt.plot([d_start_end, step_length], [end[2] ,end[2] - max_depth_change],"--", c="red")
        max_z_plot = start[2]
        min_z_plot = start[2]
        x_lim = [0, step_length * 1.4]
        for path in paths:
            path_list = np.array(path["waypoints"])
            S = path["S"]
            d_xy_list = np.linalg.norm(S[0,:2] - S[:, :2], axis=1)
            z_list = S[:, 2]
            plt.scatter(d_xy_list, z_list, c="b")
            plt.plot([0, step_length], [start[2], z_list[-1]], c="b")
            plt.xlabel("Distance [m]")
            plt.ylabel("Depth [m]")
            plt.gca().invert_yaxis()

            if max(z_list) > max_z_plot:
                max_z_plot = max(z_list)
            if min(z_list) < min_z_plot:
                min_z_plot = min(z_list)
        plt.ylim([min_z_plot - 5, max_z_plot + 5])
        plt.xlim(x_lim)
        plt.title("Possible paths")
        plt.legend()
        plt.savefig("figures/tests/PathPlanner/possible_paths.png")
        plt.close()

    def plot_consecutive_z_paths(self, path_planner):

        if self.print_while_running:
            print("Plotting consecutive z paths")

        start = np.array([np.random.uniform(0,200), np.random.uniform(0,200), np.random.uniform(70,80)])
        end = np.array([np.random.uniform(600,1500), np.random.uniform(600,1500), np.random.uniform(0,5)])
        d_start_end = np.linalg.norm(end[:2] - start[:2])
        step_length = np.random.uniform(80, 150) # m
        path_planner.step_length = step_length
        current_point = start
        depth_lim = np.random.uniform(5, 13)
        max_z_plot = start[2]
        min_z_plot = start[2]

        n_paths = np.random.randint(4, 9)
        path_planner.n_depths = n_paths

        d_b_c = d_start_end - step_length
        max_depth_change = d_b_c * np.tan(path_planner.attack_angle * np.pi / 180)
        plt.plot([d_start_end, step_length], [end[2] ,end[2] + max_depth_change],"--", c="red", label="Target depth limits")
        plt.plot([d_start_end, step_length], [end[2] ,end[2] - max_depth_change],"--", c="red")

        plt.axhline(y=0, color='g', linestyle='--', label="Field depth")
        plt.axhline(y=50, color='g', linestyle='--')

        # We move towards the end point in steps of step_length
        while np.linalg.norm(current_point - end) > 20:
            d_from_start = np.linalg.norm(current_point[:2] - start[:2])
            path_depth_limits = [current_point[2] - depth_lim, current_point[2] + depth_lim]
            paths = path_planner.get_possible_paths_z(current_point, end, path_depth_limits)
            # Choose on path at random
            path = paths[np.random.randint(0, len(paths))]
            current_point = path["waypoints"][-1]
            
            for p in paths:
                path_list = np.array(p["waypoints"])
                S = p["S"]
                d_xy_list = np.linalg.norm(start[:2] - S[:, :2], axis=1)
        
                z_list = S[:, 2]
                plt.plot([d_xy_list[0], d_xy_list[-1]], [z_list[0], z_list[-1]], c="b")
                plt.xlabel("Distance [m]")
                plt.ylabel("Depth [m]")

                if max(z_list) > max_z_plot:
                    max_z_plot = max(z_list)
                if min(z_list) < min_z_plot:
                    min_z_plot = min(z_list)
            
            # Plot the chosen path
            path_list = np.array(path["waypoints"])
            S = path["S"]
            d_xy_list = np.linalg.norm(start[:2] - S[:, :2], axis=1)
            d_waypoints = np.linalg.norm(path_list[0,:2] - path_list[:, :2], axis=1)
            z_waypoints = path_list[:, 2]
            z_list = S[:, 2]
            plt.plot([d_xy_list[0], d_xy_list[-1]], [z_list[0], z_list[-1]], c="r")
            plt.scatter(d_xy_list[-1], z_list[-1], c="g")

        plt.ylim([min_z_plot - 2, max_z_plot + 2])

        # Add the start and end point
        plt.plot(0, start[2], 'o',c="red", label="Start")
        plt.plot(d_start_end, end[2], 'o',c="red", label="Target")
        plt.legend()



        plt.savefig("figures/tests/PathPlanner/consecutive_z_paths.png")
        plt.close()

    def plot_suggest_directions(self, path_planner):
        a1 = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(0,50)])
        random_vec = np.random.uniform(-1, 1, 3)
        a2 = a1 + random_vec
        path_length = 1000

        start = a1

        end_points = path_planner.suggest_directions(start, path_length)

        plt.plot([a1[0], a2[0]], [a1[1], a2[1]], label="Previous path")
        for end_point in end_points:
            plt.plot([a2[0], end_point[0]], [a2[1], end_point[1]],c="green", label="Suggested path")
        
        # plot border of the field
        plt.axhline(y=0, color='k', linestyle='--', label="Field border")  
        plt.axhline(y=2000, color='k', linestyle='--')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axvline(x=2000, color='k', linestyle='--')
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        
        plt.savefig("figures/tests/PathPlanner/suggest_directions.png")
        plt.close()


    def plot_suggest_yoyo_paths(self, path_planner):
        start = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), 0])
        start_t = np.random.uniform(0, 24 * 3600)

        yoyo_paths = path_planner.suggest_yoyo_paths(start, start_t)

        for path in yoyo_paths:
            path_list = np.array(path["S"])
        
            z_list = path_list[:, 2]
            plt.scatter(path_list[:, 0], path_list[:,1], c=z_list, label="Observe points")

        plt.xlabel("y () [m]")
        plt.ylabel("x [m]")
        plt.title("Yo-yo paths")
        plt.legend()
        plt.savefig("figures/tests/PathPlanner/yoyo_paths.png")
        plt.close()

    def test_random_mission_path(self, path_planner):
        # Start with two yo-yo paths
        start = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), 0])
        current_wp = start
        start_t = np.random.uniform(0, 24 * 3600)
        current_t = start_t
        
        full_path = []
        for i in range(2):
            paths = path_planner.suggest_next_paths(current_wp, current_t, mode="yoyo")

            # Choose one path at random
            path = paths[np.random.randint(0, len(paths))]
            full_path.append(path)
            current_wp = path["waypoints"][-1]
            current_t = path["T"][-1]


        # Addaptive depth path
        for i in range(3):
            directions = path_planner.suggest_directions(current_wp, 1000)
            end_point = directions[np.random.randint(0, len(directions))]
            
            while np.linalg.norm(current_wp - end_point) > 20:
                paths = path_planner.suggest_next_paths(current_wp, current_t,s_target=end_point, mode="depth_seeking")
                path = paths[np.random.randint(0, len(paths))]
                full_path.append(path)
                current_wp = path["waypoints"][-1]
                current_t = path["T"][-1]

        ax = plt.figure().add_subplot(projection='3d')

        for path in full_path:
            waypoints = np.array(path["waypoints"])
            ax.plot(waypoints[:, 1], waypoints[:, 0],-waypoints[:, 2], c="b")
            ax.scatter(waypoints[0, 1], waypoints[0, 0], -waypoints[0, 2], c="g")

        plt.xlabel("(east) [m]")
        plt.ylabel("(North) [m]")
        ax.set_zlabel("Depth [m]")
        plt.title("Random mission path")
        plt.savefig("figures/tests/PathPlanner/random_mission_path_3d.png")
        plt.close()

        # Plot in 2d

        for path in full_path:
            s = path["S"]
            plt.scatter(s[:, 1], s[:, 0], c=s[:, 2], vmin=0, vmax=path_planner.auv_depth_range[1], cmap="viridis")
        
        plt.xlabel("(east) [m]")
        plt.ylabel("(North) [m]")
        plt.title("Random mission path")
        plt.savefig("figures/tests/PathPlanner/random_mission_path_2d.png")
        plt.close()


    def plot_random_paths(self, path_planner):
        
        # Start in a random point
        start = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), 0])
        start_t = np.random.uniform(0, 24 * 3600)
        current_wp = start
        current_t = start_t
        full_path = []
        for i in range(20):
            dist = np.random.uniform(100, 400)
            path = path_planner.get_random_path(current_wp, current_t, distance = dist)
            full_path.append(path)
            current_wp = path["waypoints"][-1]
            current_t = path["T"][-1]
        
        ax = plt.figure().add_subplot(projection='3d')

        for path in full_path:
            waypoints = np.array(path["waypoints"])
            ax.plot(waypoints[:, 1], waypoints[:, 0],-waypoints[:, 2], c="b")
            ax.scatter(waypoints[0, 1], waypoints[0, 0], -waypoints[0, 2], c="g")
        
        plt.xlabel("(east) [m]")
        plt.ylabel("(North) [m]")
        ax.set_zlabel("Depth [m]")
        plt.title("Random paths")
        plt.savefig("figures/tests/PathPlanner/random_paths_3d.png")
        plt.close()

    
    def plot_probable_path(self, path_planner):

        start = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), 20])
        start_t = np.random.uniform(0, 24 * 3600)

        path_straight = path_planner.get_random_path(start, start_t, distance=200)
        end = path_straight["waypoints"][-1]
        print(path_straight["waypoints"])

        path = path_planner.get_probable_path(start, end, start_t)

        print(path)
        print(path_straight)

        dist_path_straight = np.linalg.norm(path_straight["S"][0] - path_straight["S"])
        dist_path = np.linalg.norm(path["S"][0] - path["S"])

        plt.plot(dist_path, path["S"][:, 2], label="Probable path")
        plt.plot(dist_path_straight, path_straight["S"][:, 2], label="Straight path")
        plt.title("Probable path vs straight path")
        plt.savefig("figures/tests/PathPlanner/probable_path_v_straight_path.png")
        plt.close()
        
    def plot_suggest_paths(self, path_planner):

        s = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), 30])
        t = np.random.uniform(0, 24 * 3600)

        time_remaining_submerged = 300

        attack_angle = path_planner.attack_angle
        speed = path_planner.max_speed
        max_dist = speed * time_remaining_submerged 
        max_depth_change = max_dist * np.tan(attack_angle / 180 * np.pi)
        print(max_depth_change)
        step_length = 100
        
        n_directions = 5
        n_depths = 5
        paths = path_planner.suggest_paths(s, t, time_remaining_submerged,
                                            step_length = step_length,
                                            n_depths=n_depths,
                                            n_directions=n_directions)

        tot_points = 0
        # plot the paths in 3d
        ax = plt.figure().add_subplot(projection='3d')
        for path in paths:
            waypoints = np.array(path["waypoints"])
            #print(path)
            #print(waypoints)
            #ax.plot(waypoints[0], waypoints[0, 0],-waypoints[:, 2], c="b")
            #ax.plot(waypoints[0], waypoints[-1])
            S = path["S"]
            tot_points += len(S)
            ax.scatter(S[:, 1], S[:, 0], -S[:, 2], c=S[:, 2], cmap="viridis")
        print("total points", tot_points)
        plt.xlabel("(east) [m]")
        plt.ylabel("(North) [m]")
        ax.set_zlabel("Depth [m]")
        plt.title("Suggested paths")
        plt.savefig("figures/tests/PathPlanner/suggested_paths_3d.png")
        plt.show()


    def plot_suggest_paths_multi_step(self, path_planner):

        s = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), 30])
        t = np.random.uniform(0, 24 * 3600)
        step_length =  250
        
        n_directions = 5
        n_depths = 5
        time_remaining_submerged = np.random.uniform(2 * step_length, 3.5 * step_length)

        n_steps = min(int(time_remaining_submerged / step_length) , 3)

        attack_angle = path_planner.attack_angle
        speed = path_planner.max_speed
        max_dist = speed * time_remaining_submerged 
        max_depth_change = max_dist * np.tan(attack_angle / 180 * np.pi)
        print(max_depth_change)
        
        paths = path_planner.suggest_multi_step_paths(s, t, time_remaining_submerged,
                                            n_steps = n_steps,
                                            step_length = step_length,
                                            n_depths=n_depths,
                                            n_directions=n_directions)
        

        print(len(paths))
        tot_points = 0
        unique_points = np.empty((0, 4))
        # plot the paths in 3d
        ax = plt.figure().add_subplot(projection='3d')
        for path in paths:
            waypoints = np.array(path["waypoints"])
            T = path["T"]
            tot_points += len(T)
            # Plot the path
            ax.plot(waypoints[:, 1], waypoints[:, 0],-waypoints[:, 2], c="b")
        print("total points", tot_points)
        plt.xlabel("(east) [m]")
        plt.ylabel("(North) [m]")
        ax.set_zlabel("Depth [m]")
        plt.title("Suggested paths")
        plt.savefig("figures/tests/PathPlanner/suggested_paths_multi_step_3d.png")
        plt.show()






