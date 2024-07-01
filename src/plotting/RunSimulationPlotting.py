import matplotlib.pyplot as plt
import numpy as np

class RunSimulationPlotting:
    def __init__(self, run_simulation):
        self.run_simulation = run_simulation

    def plot_depth_seek_one_step(self, path_evaluation_data, name=None):

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        paths = path_evaluation_data["paths"]
        best_design = path_evaluation_data["best_design"]
        design_data = path_evaluation_data["design_data"]
        best_design_id = path_evaluation_data["best_id"]
        best_path = paths[best_design_id]

        start_s = best_path["S"][0, :]
        start_t = best_path["T"][-1]

        max_z, min_z = start_s[2], start_s[2]

        for i, path in enumerate(paths):

            path_list = np.array(path["waypoints"])
            d_xy_list = np.linalg.norm(path_list[0,:2] - path_list[:, :2], axis=1)
            z_list = path_list[:, 2]
            ax[0].plot(d_xy_list, z_list, c="grey", alpha=0.3, label="Waypoints")
            ax[0].plot(d_xy_list[-1], path_list[-1,2],'o', c="blue", label="End")
            ax[0].set_xlabel("Distance [m]")
            ax[0].set_ylabel("Depth [m]")

            max_z = max(max_z, max(z_list))
            min_z = min(min_z, min(z_list))

            if i == best_design_id:
                ax[0].plot(d_xy_list, z_list, c="red", label="Best path")
                ax[0].plot(d_xy_list[-1], z_list[-1],'o', c="red", label="Best end")
        
        end_yx = best_path["S"][-1, :2]
        end_t = best_path["T"][-1]

        
        k = 20
        pred_y = np.repeat(end_yx[0], k)
        pred_x = np.repeat(end_yx[1], k)
        pred_z = np.linspace(min_z, max_z, k)
        S_pred = np.array([pred_y, pred_x, pred_z]).T
        T_pred = np.repeat(end_t, k)
        mu_pred, Sigma_pred = self.run_simulation.model.predict(S_pred, T_pred)
        mu_true = self.run_simulation.field.get_intensity_x(S_pred, T_pred)
        ax[1].plot(mu_pred, pred_z, label="Predicted intensity")
        ax[1].plot(mu_true, pred_z, label="True intensity")
        ax[1].fill_betweenx(pred_z, mu_pred - 2 * np.sqrt(np.diag(Sigma_pred)), mu_pred + 2 * np.sqrt(np.diag(Sigma_pred)),  alpha=0.2)
        ax[1].set_xlabel("Intensity")
        ax[1].set_ylabel("Depth [m]")
        ax[1].legend()

        if name is not None:
            plt.savefig("figures/tests/RunSimulation/depth_seek_one_step_" + name + ".png")
        else:
            plt.savefig("figures/tests/RunSimulation/depth_seek_one_step.png")
        plt.close()
    
    def plot_depth_seeker(self,end_point, name=None):
        """
        Here we move towards the end point in several steps, in each step we move in the direction of the end point
        but choose a different depth. The depth is chosen to maximize the expected intensity and end up in the correct depht
        at the end.
        """

        fig, ax = plt.subplots(3, 1, figsize=(15,10))
        # Firsth we find the where the AUV is now
        start_s = self.run_simulation.model.get_s_now()
        start_t = self.run_simulation.model.get_t_now()
        s_now = start_s
        t_now = start_t

        d_start_end = np.linalg.norm(start_s[:2] - end_point[:2])
        depth_lim = 10
        step_length = self.run_simulation.path_planner.step_length



        ax[0].axhline(y=0, color='g', linestyle='--', label="Field depth")
        ax[0].axhline(y=50, color='g', linestyle='--')

        # Plot start and end point
        ax[0].plot(0, start_s[2], 'o', c="blue", label="Start")
        ax[0].plot(d_start_end, end_point[2], 'o', c="blue", label="End")

        min_z, max_z = start_s[2], start_s[2]
        min_d, max_d = 0,d_start_end
        
        i_step = 0
        while np.linalg.norm(s_now - end_point) > 20:
            # Find the best path at this point in time
            path_evaluation_data = self.run_simulation.depth_seeking_survey_step(end_point)

            paths = path_evaluation_data["paths"]
            best_design = path_evaluation_data["best_design"]
            design_data = path_evaluation_data["design_data"]
            best_design_id = path_evaluation_data["best_id"]
            best_path = paths[best_design_id]

            

            print(i_step)
            i_step += 1
            loc_z_min = max(min([p["S"][-1, 2] for p in paths]) - 5, 0)
            loc_z_max = max([p["S"][-1, 2] for p in paths]) + 5
            yx_next_step = best_path["S"][-1, :2]
            t_next_step = best_path["T"][-1]
            n_vert = 30
            pred_x = np.repeat(yx_next_step[1], n_vert)
            pred_y = np.repeat(yx_next_step[0], n_vert)
            pred_z = np.linspace(loc_z_min, loc_z_max, n_vert)
            S_pred = np.array([pred_y, pred_x, pred_z]).T
            T_pred = np.repeat(t_next_step, n_vert)
            prior_mu, prior_Sigma = self.run_simulation.model.predict(S_pred, T_pred)
            true_mu = self.run_simulation.field.get_intensity_x(S_pred, T_pred)

            # Plot the path
            for i, path in enumerate(paths):
                path_list = np.array(path["S"])
                d_xy_list = np.linalg.norm(start_s[:2] - path_list[:, :2], axis=1)
                z_list = path_list[:, 2]
                if i == 0 and i_step == 1:
                    ax[0].plot(d_xy_list, z_list, c="grey", alpha=0.3, label="Waypoints")
                else:
                    ax[0].plot(d_xy_list, z_list, c="grey", alpha=0.3)
                

                max_z = max(max_z, max(z_list))
                min_z = min(min_z, min(z_list))

                if i == best_design_id:
                    if i_step == 1:
                        ax[0].plot(d_xy_list, z_list, c="red", label="Best path")
                        ax[0].scatter(d_xy_list, z_list, c="black", s=4, label="Data points")
                        ax[0].plot(d_xy_list[-1], z_list[-1],'o', c="red", label="Best end")
                    else:
                        ax[0].plot(d_xy_list, z_list, c="red")
                        ax[0].scatter(d_xy_list, z_list, c="black",s=4)
                        ax[0].plot(d_xy_list[-1], z_list[-1],'o', c="red")

                    ax[1].plot(d_xy_list, z_list, c="black", alpha=0.3)
                    ax[1].plot(d_xy_list[-1], z_list[-1],'o', c="black",alpha=0.3)

            # Move the agent to the next location
            self.run_simulation.move_agent(best_path)

            # Plot the new data points 
            n_new_data_points = len(best_path["S"])
            





            posterior_mu, posterior_Sigma = self.run_simulation.model.predict(S_pred, T_pred)

            mu_min = min(min(prior_mu), min(true_mu), min(posterior_mu))
            mu_max = max(max(prior_mu), max(true_mu), max(posterior_mu))
            mu_prior_scaled = (prior_mu - mu_min) / (mu_max - mu_min) * step_length * 0.8 + i_step * step_length
            mu_true_scaled = (true_mu - mu_min) / (mu_max - mu_min) * step_length * 0.8 + i_step * step_length
            mu_posterior_scaled = (posterior_mu - mu_min) / (mu_max - mu_min) * step_length * 0.8 + i_step * step_length

            d_max = max(max(mu_prior_scaled), max(mu_true_scaled), max(mu_posterior_scaled))

            if i_step == 1:
                ax[1].plot(mu_prior_scaled, pred_z, c="blue", label="Forcast intensity")
                ax[1].plot(mu_true_scaled, pred_z, c="green", label="True intensity")
                ax[1].plot(mu_posterior_scaled, pred_z,c="red", label="Posterior intensity")
            else:
                ax[1].plot(mu_prior_scaled, pred_z, c="blue")
                ax[1].plot(mu_true_scaled, pred_z, c="green")
                ax[1].plot(mu_posterior_scaled, pred_z,c="red")
            # Add verticle line
            ax[1].axvline(x=step_length*i_step, color='grey', linestyle='--')

            # Add the predictions to the plot
            x_pred_best = best_design["x_pred"]
            S_pred_best = best_design["s"]
            d_s_best = np.linalg.norm(S_pred_best[:,:2] - start_s[:2], axis=1)
            Sigma_pred = best_design["Sigma_pred"]
            dSigma_pred = np.sqrt(np.diag(Sigma_pred))
            if i_step == 1:
                ax[2].plot(d_s_best, x_pred_best, c="blue", label="Predicted intensity")
            else:
                ax[2].plot(d_s_best, x_pred_best, c="blue")
            ax[2].fill_between(d_s_best, x_pred_best - 1.965 * dSigma_pred, x_pred_best + 1.965 * dSigma_pred, alpha=0.2, color="blue")


            # Update the current location
            s_now = self.run_simulation.model.get_s_now()
            t_now = self.run_simulation.model.get_t_now()

        # Add the fitted model along the path
        s_obs, t_obs = self.run_simulation.model.get_all_observations_st()
        y_obs = self.run_simulation.model.get_all_observations_y()
        x_obs = self.run_simulation.model.get_all_observations_x()
        x_true = self.run_simulation.field.get_intensity_x(s_obs, t_obs)
        dSigma = np.sqrt(self.run_simulation.model.get_all_dSigma())
        d_obs = np.linalg.norm(s_obs[:,:2] - start_s[:2], axis=1)
 
        ax[2].scatter(d_obs, np.log(y_obs + 1),marker="x", alpha=0.5, c="black", label=r"ln($y_{obs}$ + 1)")
        ax[2].plot(d_obs, x_obs, c="red", label=r"$x_{obs}$")
        ax[2].plot(d_obs, x_true, c="green", label=r"$x_{true}$")
        ax[2].fill_between(d_obs, x_obs - 1.965 * np.sqrt(dSigma), x_obs + 1.965 * np.sqrt(dSigma), alpha=0.2, color="red")


        ax[0].set_title("Path taken by the Agent")
        ax[1].set_title("Intensity in depth profile")
        ax[2].set_title("Intensity along the path taken by the Agent")

        # Axis names
        ax[0].set_xlabel("Distance [m]")
        ax[0].set_ylabel("Depth [m]")
        ax[1].set_xlabel("Distance [m]")
        ax[1].set_ylabel("Depth [m]")
        ax[2].set_xlabel("Distance [m]")
        ax[2].set_ylabel("Intensity")
        ax[0].set_xlim([0, d_max])
        ax[1].set_xlim([0, d_max])
        ax[2].set_xlim([0, d_max])

        ax[1].set_ylim([min_z, max_z])
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        if name is not None:
            plt.savefig("figures/tests/RunSimulation/depth_seeker_" + name + ".png")
        else:
            plt.savefig("figures/tests/RunSimulation/depth_seeker.png")
        plt.close()


    def plot_path_auv(self, name=None):
        """
        Plot the path taken by the AUV
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        s_path, t_path = self.run_simulation.model.get_all_observations_st()

        ax.scatter(s_path[:,0], s_path[:,1], -s_path[:,2], c=t_path, label="Observations")
        ax.set_zlabel("Depth [m]")
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.legend()
        if name is not None:
            plt.savefig("figures/tests/RunSimulation/path_auv_" + name + ".png")
        else:
            plt.savefig("figures/tests/RunSimulation/path_auv.png")
        plt.close()
        
    def plot_path_auv_observation(self, name=None):
        """
        Plot the path taken by the AUV
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        s_path, t_path = self.run_simulation.model.get_all_observations_st()
        x_obs = self.run_simulation.model.get_all_observations_x()

        ax.scatter(s_path[:,0], s_path[:,1], -s_path[:,2], c=x_obs, label="Observations")
        ax.set_zlabel("Depth [m]")
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.legend()
        if name is not None:
            plt.savefig("figures/tests/RunSimulation/path_auv_x_" + name + ".png")
        else:
            plt.savefig("figures/tests/RunSimulation/path_auv_x.png")
        plt.close()
        
