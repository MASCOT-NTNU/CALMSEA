import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import time
import pickle
import datetime
#import seaborn as sns # REMOVE THIS LATER
from help_func import *
import matplotlib.pyplot as plt # Remove this later, only for debugging

from Model import Model
 
class ModelLogNormal(Model):

    def __init__(self, 
                 prior,
                 covariance_funtion_distance_str: str = 'matern',
                 covariance_funtion_temporal_str: str = 'exponential',
                 phi_z: float = 0.5,
                 phi_yx: float = 1 / 1000,
                 phi_temporal: float = 0.7 / 3600,
                 sigma: float = 0.7,
                 tau: float = 1,

                 print_while_running: bool = False,
                 print_warnings: bool = True,
                 ):
        
        ## setting the model
        self.prior = prior
        #self.prior_mean_function = prior_mean_function
        


        ## setting parameters for the mode
        self.model_parameters = {"covariance_funtion_distance_str": covariance_funtion_distance_str,
                                "covariance_funtion_temporal_str": covariance_funtion_temporal_str,
                                "phi_z": phi_z,
                                "phi_yx": phi_yx,
                                "phi_temporal": phi_temporal,
                                "sigma": sigma,
                                "tau": tau}
        


        # Data storage
        self.data_dict = {}
        self.data_dict["has_data"] = False
        self.all_data_dict = {}
        self.all_data_dict["has_data"] = False
        self.convergence_data = {} # Data about the convergence of the algorithm
        self.convergence_data["norm_list"] = []
        self.convergence_data["iterations"] = []
        self.convergence_data["time"] = []
        self.convergence_data["n_points"] = []
        self.timing = True # If the timing data should be stored
        self.timing_data = {} # Time the different parts of the algorithm

        # Print while running
        self.print_warnings = print_warnings  # These are where there is some issue with the model
        self.print_while_running = print_while_running


    def add_first_values(self, S: np.ndarray,  T: np.ndarray, Y: np.ndarray):
        # Add the first values to the model
        # S: spatial points
        # Y: observations
        # T: temporal points

        # Get the prior mean
        #mu = self.prior_mean_function(S, T)
        mu = self.prior.get_prior_S_T(S, T)

        # Get the covariance matrix
        Sigma = self.make_covariance_matrix(S, T)

        # Save the data
        self.data_dict["has_data"] = True
        self.data_dict["S"] = S
        self.data_dict["T"] = T
        self.data_dict["y"] = Y
        self.data_dict["mu"] = mu
        self.data_dict["Sigma"] = Sigma
        self.data_dict["batch"] = np.repeat(0, len(Y))

        # Fit the model 
        x, Sigma = self.fit(mu, Y, Sigma)

        # Save the fitted data
        self.data_dict["x"] = x
        return x, Sigma

    def filter_values(self, S: np.ndarray, T: np.array, Y: np.array):

        if self.print_while_running:
            print(time_now_str(), f"[ACTION] [MODEL] filtering the datapoints")
        
        Y = Y + 0.0001
        return S, T, Y


    def add_new_values(self, S_new: np.ndarray, T_new: np.array, Y_new: np.array):
        # Add new values to the model
        # S_new: new spatial points
        # Y_new: new observations
        # T_new: new temporal points

        S_new, T_new, Y_new = self.filter_values(S_new, T_new, Y_new)
        start_t = time.time()     # Start time
        if self.print_while_running:
            print(time_now_str(), f"[ACTION] Adding {len(T_new)} new values to the model")   

        if len(Y_new) == 0:
            if self.print_warnings:
                print(time_now_str(), "[WARNING] [MODEL] No new values to add to the model")
            return

        # See how many points are added 
        n_new = len(Y_new)

        if self.data_dict["has_data"] == False:
            self.add_first_values(S_new, T_new, Y_new)
        else:
            
            # Load data from memory
            s_old = self.data_dict["S"]
            t_old = self.data_dict["T"]
            y_old = self.data_dict["y"]
            mu_old = self.data_dict["mu"]
            Sigma_old = self.data_dict["Sigma"]
            x_old = self.data_dict["x"]


            # Get the dimensions of the old and new data
            n_old, n_new = len(y_old), len(Y_new)


            # Join the old and new data
            s = np.concatenate((s_old, S_new))
            t = np.concatenate((t_old, T_new))
            y = np.concatenate((y_old, Y_new))
            #mu_new = self.prior_mean_function(S_new, T_new) 
            mu_new = self.prior.get_prior_S_T(S_new, T_new)
            mu = np.concatenate((mu_old, mu_new))

            # Get the cross covariance matrix
            # This can be done more effieciently
            Sigma = self.make_covariance_matrix(s, t)

            
            # Fit the model 
            x, Sigma = self.fit(mu, y, Sigma)

          
            # Save the data
            self.data_dict["has_data"] = True
            self.data_dict["S"] = s
            self.data_dict["T"] = t
            self.data_dict["y"] = y
            self.data_dict["mu"] = mu
            self.data_dict["Sigma"] = Sigma
            self.data_dict["x"] = x

            # Add the batch number
            batch_num = np.max(self.data_dict["batch"]) + 1
            self.data_dict["batch"] = np.concatenate((self.data_dict["batch"], np.repeat(batch_num, n_new)))
            


        # Update the all_data_dict
        self.update_all_data(n_new)
        
        end_t = time.time() 
        self.update_timing("add_new_values", end_t - start_t)


    def fit(self,mu, y_obs, Sigma):
        """
        Fit the model to the data

        mu: prior mean
        y_obs: observed values
        Sigma: Covariance matrix
        tau: The standard deviation of the observation noise
        """
        t1 = time.time()        # Start time

        tau = self.model_parameters["tau"]
        P = np.eye(len(y_obs)) * tau**2
        # TODO: Implement better inversion here
        Sigma_P_inv = np.linalg.inv(Sigma + P)
        Sigma_at_Sigma_P_inv = Sigma @ Sigma_P_inv
        x = mu + Sigma @ Sigma_P_inv @ (np.log(y_obs) - mu)
        Sigma_cond = Sigma - Sigma_at_Sigma_P_inv @ Sigma

        self.data_dict["Sigma_P_inv"] = Sigma_P_inv

        t2 = time.time()            # End time
        if self.print_while_running:
            print(time_now_str(), f"[INFO] Fitting the model took {t2 - t1:.2f} seconds")
        self.update_timing("fit", t2 - t1)
        return x, Sigma_cond
    

    def predict(self, S_B: np.ndarray, T_B: np.ndarray):

        start_t = time.time()    # Start time   
        if self.data_dict["has_data"] == False:
            if self.print_while_running:
                print(time_now_str() ,"[INFO] [MODEL] No data in model, predicting based on prior mean function")

            #mu_B = self.prior_mean_function(S_B, T_B)
            mu_B = self.prior.get_prior_S_T(S_B, T_B)
            Sigma_BB = self.make_covariance_matrix(S_B, T_B)
            return mu_B, Sigma_BB
        
        """
        Predict x_B based on x_A_est and Sigma_est
        s_B: The points to predict
        s_A: The points used for estimating x_A_est
        x_A_est: The estimated x_A
        Sigma_est: The estimated covariance matrix of x_A_est
        """

        # Load data from memory
        S_A = self.data_dict["S"]
        T_A = self.data_dict["T"]
        y = self.data_dict["y"]
        Sigma_AA_est = self.data_dict["Sigma"]

        # Get the cross covariance matrix
        Sigma_AB = self.make_covariance_matrix_2(S_B, S_A, T_B, T_A)
        Sigma_BA = Sigma_AB.T

 
        Sigma_BB = self.make_covariance_matrix(S_B, T_B)               # Get the covariance matrix of x_B
        Sigma_AA = self.make_covariance_matrix(S_A, T_A)               # This should already be calculated, but it does not take a lot of time 

        # Get mean of x_B and x_A
        mu_A = self.data_dict["mu"]
        #mu_B = self.prior_mean_function(S_B, T_B)     
        mu_B = self.prior.get_prior_S_T(S_B, T_B)
                    

        # Get conditional mean and covariance matrix
        Sigma_AAP_inv = self.data_dict["Sigma_P_inv"]  
        x_B_est = mu_B + Sigma_AB @ Sigma_AAP_inv @ (np.log(y) - mu_A)
        Sigma_BB_est = Sigma_BB - Sigma_AB @ Sigma_AAP_inv @ Sigma_BA


        end_t = time.time()             # End time
        self.update_timing("predict", end_t - start_t)                  # Update the timing data
        return x_B_est, Sigma_BB_est
    

    def down_sample_points(self):
        if self.print_while_running:
            print(time_now_str(), "[ACTION] Down sampling points")


        t_start = time.time()

        # This function removes half of the points in the data
        # This is done to save time

        old_data = self.data_dict
        new_data = {}
        self.data_dict = {}

        # TODO: can add a smarter way to add these indecies
        ind = [True if i % 2 == 0 else False for i in range(len(old_data["S"]))]


        # New data
        self.data_dict["x"] = old_data["x"][ind]
        self.data_dict["y"] = old_data["y"][ind]
        self.data_dict["S"] = old_data["S"][ind]
        self.data_dict["T"] = old_data["T"][ind]
        self.data_dict["Sigma"] = old_data["Sigma"][ind][:,ind]
        self.data_dict["mu"] = old_data["mu"][ind]
        self.data_dict["batch"] = old_data["batch"][ind]
        self.data_dict["has_data"] = False # This will be true after the fit

        # Refit the model
        cov_mat = self.make_covariance_matrix(self.data_dict["S"], self.data_dict["T"])
        self.data_dict["x"], self.data_dict["Sigma"] = self.fit(self.data_dict["mu"], self.data_dict["y"], cov_mat)

        # Adding the unchanged data
        self.data_dict["has_data"] = True #old_data["has_data"]
        
        t_end = time.time()

        # Store timing
        self.update_timing("down_sample_points", t_end - t_start)
        if self.print_while_running:
            print(time_now_str(), "[ACTION] [MODEL] Down sampling points done")
            print(time_now_str(), "[TIMING] [MODEL] \t Time for downsampling: ", round(t_end - t_start,3), " s")




if __name__ == '__main__':

    #######################################################################
    #######################################################################
    ############    TESTING THE MODEL    #################################
    #######################################################################
    #######################################################################
    import matplotlib.pyplot as plt

    from FieldLogNormal import FieldLogNormal
    from Prior import Prior

    figures_path = "figures/tests/ModelLogNormal/"

    prior = Prior(parameters={"beta_0":0.5,
                               "beta_1":2,
                               "beta_2":1 / 6**2,
                               "peak_depth_min":10,
                               "peak_depth_max":60})

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
    
    # Test the model
    model = ModelLogNormal(prior=prior, print_while_running=True)

    time_1 = 24 * 3600 / 2

    # Get the field
    field = FieldLogNormal(print_while_running=True)
    # Generate the field
    n_s = [15,15,15] # 
    n_t = 15 # 
    S_lim = [[0,2000],[0,2000],[0,50]] # 2000 meters in the x and y direction, 50 meters in the z direction
    T_lim = [0,24 * 3600] # 24 hours
    field.generate_field_x(S_lim,T_lim,n_s,n_t)

    # Get some observations
    Sz = np.linspace(0,25,100)
    sy = np.repeat(1000,100)
    Sx = np.repeat(1000,100)
    S = np.column_stack((Sx,sy,Sz))
    T = np.repeat(time_1 ,100)
    Y = field.get_observation_Y(S,T)

    Sz_pred = np.linspace(0,50,100)
    sy_pred = np.repeat(1000,100)
    Sx_pred = np.repeat(1000,100)
    S_pred = np.column_stack((Sx_pred,sy_pred,Sz_pred))
    T_pred = np.repeat(time_1 ,100)

    # Add the first values to the model
    model.add_new_values(S,T,Y)
    pred_x, pred_Sigma = model.predict(S_pred, T_pred)

    # Plot the estimated intensity
    x_true = field.get_intensity_x(S_pred,T_pred)
    S = model.data_dict["S"]
    T = model.data_dict["T"]
    mu = model.data_dict["mu"]
    Sigma = model.data_dict["Sigma"]
    plt.plot(S_pred[:,2],pred_x, label="Estimated intensity")
    plt.scatter(S[:,2], np.log(Y), label="ln(Observations)", color="red", marker="x", alpha=0.5)
    plt.plot(S[:,2],x_true, label="True intensity")
    plt.plot(S[:,2],mu, label="Prior mean")
    plt.plot(S[:,2],pred_x + 1.965 * np.sqrt(np.diag(Sigma)), label="95% confidence interval",color="blue", linestyle="--")
    plt.plot(S[:,2],pred_x - 1.965 * np.sqrt(np.diag(Sigma)), linestyle="--", color="blue")
    plt.legend()
    plt.savefig("figures/tests/ModelLogNormal/First_values" + ".png")
    plt.close()

    # Predict some more values 
    Sz_new = np.linspace(26,50,97)
    Sy_new = np.repeat(1000,97)
    Sx_new = np.repeat(1000,97)
    print("Sx_new shape", Sx_new.shape)
    print("Sy_new shape", Sy_new.shape)
    print("Sz_new shape", Sz_new.shape)
    S_new = np.column_stack((Sx_new,Sy_new,Sz_new))
    print("S_new shape", S_new.shape)
    T_new = np.repeat(time_1 ,97)
    print("Is this the break?") # REMOVE
    x_B_est, Sigma_BB_est = model.predict(S_new, T_new)
    print("Did we survive?") # REMOVE

    # Plot the estimated intensity
    x_est = model.data_dict["x"]
    x_true = field.get_intensity_x(S,T)
    S = model.data_dict["S"]
    T = model.data_dict["T"]
    mu = model.data_dict["mu"]
    Sigma = model.data_dict["Sigma"]
    plt.plot(S[:,2],x_est, label="Estimated intensity")
    plt.plot(S[:,2],x_true, label="True intensity")
    plt.plot(S[:,2],mu, label="Prior mean")
    plt.plot(S[:,2],x_est + 1.965 * np.sqrt(np.diag(Sigma)), label="95% confidence interval",color="blue", linestyle="--")
    plt.plot(S[:,2],x_est - 1.965 * np.sqrt(np.diag(Sigma)), linestyle="--", color="blue")
    plt.plot()
    plt.legend()
    plt.savefig(figures_path + "/First_values" + ".png")
    plt.close()

    # Plot the exp(x)
    plt.plot(S[:,2],np.exp(x_est), label="Estimated intensity")
    plt.plot(S[:,2],np.exp(x_true), label="True intensity")
    plt.plot(S[:,2],np.exp(x_est + 1.965 * np.sqrt(np.diag(Sigma))), label="95% confidence interval", linestyle="--", color="blue")
    plt.plot(S[:,2],np.exp(x_est - 1.965 * np.sqrt(np.diag(Sigma))), linestyle="--", color="blue")
    plt.scatter(S[:,2],Y, label="Observations", color="red", marker="x", alpha=0.5)

    plt.legend()
    plt.savefig("figures/tests/ModelLogNormal/First_values" + ".png")
    plt.close()


    # predict some points 
    Sz_new = np.linspace(25,50,97)
    Sy_new = np.repeat(1000,97)
    Sx_new = np.repeat(1000,97)
    S_new = np.column_stack((Sx_new,Sy_new,Sz_new))
    T_new = np.repeat(time_1 ,97)
    x_B_est, Sigma_BB_est = model.predict(S_new, T_new)
    #mu_new = model.prior_mean_function(S_new, T_new)
    mu_new = model.prior.get_prior_S_T(S_new, T_new)

    # Plot the estimated intensity
    x_true_new = field.get_intensity_x(S_new,T_new)
    S = model.data_dict["S"]
    T = model.data_dict["T"]
    mu = model.data_dict["mu"]
    Sigma = model.data_dict["Sigma"]
    plt.plot(S[:,2],x_est, label="Estimated intensity")
    plt.plot(S[:,2],x_true, label="True intensity")
    plt.plot(S[:,2],mu, label="Prior mean")
    plt.plot(S[:,2],x_est + 1.965 * np.sqrt(np.diag(Sigma)), label="95% confidence interval",color="blue", linestyle="--")
    plt.plot(S[:,2],x_est - 1.965 * np.sqrt(np.diag(Sigma)), linestyle="--", color="blue")
    plt.plot(S_new[:,2],x_B_est, label="Estimated intensity")
    plt.plot(S_new[:,2],x_true_new, label="True intensity")
    plt.plot(S_new[:,2],mu_new, label="Prior mean")
    plt.plot(S_new[:,2],x_B_est + 1.965 * np.sqrt(np.diag(Sigma_BB_est)), label="95% confidence interval",color="red", linestyle="--")
    plt.plot(S_new[:,2],x_B_est - 1.965 * np.sqrt(np.diag(Sigma_BB_est)), linestyle="--", color="red")
    plt.legend()
    plt.savefig(figures_path + "/Predict" + ".png")
    plt.close()


    # add more values

    S_new = np.linspace([1000,1000,26],[1000,1000,50],97)
    print("S_new shape", S_new.shape)
    T_new = np.repeat(time_1 ,97)
    Y_new = field.get_observation_Y(S_new,T_new)
    model.add_new_values(S_new,T_new,Y_new)

    # Plot the estimated intensity
    x_est = model.data_dict["x"]
   
    S = model.data_dict["S"]
    T = model.data_dict["T"]
    x_true = field.get_intensity_x(S,T)
    mu = model.data_dict["mu"]
    Sigma = model.data_dict["Sigma"]
    plt.plot(S[:,2],x_est, label="Estimated intensity")
    plt.plot(S[:,2],x_true, label="True intensity")
    plt.plot(S[:,2],mu, label="Prior mean")
    plt.plot(S[:,2],x_est + 1.965 * np.sqrt(np.diag(Sigma)), label="95% confidence interval",color="blue", linestyle="--")
    plt.plot(S[:,2],x_est - 1.965 * np.sqrt(np.diag(Sigma)), linestyle="--", color="blue")
    plt.legend()
    plt.savefig(figures_path + "/Second_values" + ".png")
    plt.close()

    # Down sample the points
    print("Data shape before down sampling")
    model.print_data_shape()
    model.down_sample_points()
    print("Data shape after down sampling")
    model.print_data_shape()
    # Plot the estimated intensity
    x_est = model.data_dict["x"]
    s = model.data_dict["S"]
    t = model.data_dict["T"]
    x_true = field.get_intensity_x(s,t)
    mu = model.data_dict["mu"]
    Sigma = model.data_dict["Sigma"]
    y = model.data_dict["y"]

    plt.title("Down sampled points")
    plt.plot(s[:,2],x_est, label="Estimated intensity")
    plt.plot(s[:,2],x_true, label="True intensity")
    plt.plot(s[:,2],mu, label="Prior mean")
    plt.plot(s[:,2],x_est + 1.965 * np.sqrt(np.diag(Sigma)), label="95% confidence interval",color="blue", linestyle="--")
    plt.plot(s[:,2],x_est - 1.965 * np.sqrt(np.diag(Sigma)), linestyle="--", color="blue")
    plt.scatter(s[:,2],y, label="Observations", color="red", marker="x", alpha=0.5)
    plt.plot(s[:,2], np.exp(x_true), label="True lambda", color="black")
    plt.plot(s[:,2], np.exp(x_est), label="Predicted lambda", color="red")
    plt.legend()
    plt.savefig(figures_path + "/Down_sampling" + ".png")
    plt.close()


    # Try to move in a yo-yo pattern
    from PathPlanner import PathPlanner
    field_limits = [[0,2000], [0,2000], [0,50]] # y, x, z limits
    path_planner = PathPlanner(field_limits)
    
    start = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(10,40)])
    end = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(10,40)])
    yoyo_depth_limits = [end[2] + 10, end[2] - 10]
    path = path_planner.yo_yo_path(start, end, yoyo_depth_limits)

    S_path = path["S"]
    T_path = path["T"]
    observations = field.get_observation_Y(S_path, T_path)

    model = ModelLogNormal(prior=prior, print_while_running=True)
    model.add_new_values(S_path, T_path, observations)

    # Plot the estimated intensity
    x_est = model.data_dict["x"]
    s = model.data_dict["S"]
    t = model.data_dict["T"]
    x_true = field.get_intensity_x(s,t) 
    mu = model.data_dict["mu"]
    Sigma = model.data_dict["Sigma"]
    y = model.data_dict["y"]

    plt.title("Yo-yo pattern")
    plt.plot(s[:,2],x_est, label="Estimated intensity")
    plt.plot(s[:,2],x_true, label="True intensity")
    plt.plot(s[:,2],mu, label="Prior mean")
    plt.plot(s[:,2],x_est + 1.965 * np.sqrt(np.diag(Sigma)), label="95% confidence interval",color="blue", linestyle="--")
    plt.plot(s[:,2],x_est - 1.965 * np.sqrt(np.diag(Sigma)), linestyle="--", color="blue")
    plt.scatter(s[:,2],y, label="Observations", color="red", marker="x", alpha=0.5)
    plt.plot(s[:,2], np.exp(x_true), label="True lambda", color="black")
    plt.plot(s[:,2], np.exp(x_est), label="Predicted lambda", color="red")
    plt.legend()
    plt.savefig(figures_path +  "/Yo_yo" + ".png")
    plt.close()









  

