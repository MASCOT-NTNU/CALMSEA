import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import time
import pickle
import datetime
from help_func import *
#import seaborn as sns # REMOVE THIS LATER
import matplotlib.pyplot as plt # Remove this later, only for debugging


 
class Model:

    def __init__(self, 
                 prior, 
                 covariance_funtion_distance_str: str = 'matern',
                 covariance_funtion_temporal_str: str = 'exponential',
                 phi_z: float = 0.5,
                 phi_yx: float = 1 / 1000,
                 phi_temporal: float = 0.7 / 3600,
                 sigma: float = 0.7,

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
                                "reduce_factor": 0.5}
        


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


    def get_all_observations_y(self):
        # Get all the observations in the model
        return self.all_data_dict["y"]
    
    def get_all_observations_st(self):
        # Get all the data points in the model
        return self.all_data_dict["S"], self.all_data_dict["T"]
    
    def get_all_observations_x(self):
        # Get all the intensity values in the model
        return self.all_data_dict["x"]
    
    def get_all_dSigma(self):
        # Get all the covariance matrices in the model
        return self.all_data_dict["dSigma"]
    
    def get_time_to_add_values(self):
        # Get the time to add new values to the model
        return self.timing_data["add_values"]

    def covariance_function_temporal(self, h):
        phi_temporal = self.model_parameters["phi_temporal"]

        # The temporal covariance function
        return np.exp(-h * phi_temporal) 

    def covariance_function_z(self, h):
        phi_z = self.model_parameters["phi_z"]

        # The covariance function for the distance
        return np.exp(-h * phi_z) * (1 + h * phi_z) 
    
    def covariance_function_yx(self, h):
        phi_yx = self.model_parameters["phi_yx"]

        # The covariance function for the distance
        return np.exp(-h * phi_yx) * (1 + h * phi_yx)

    def time_now_str(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_likelihood_y_given_x(self, x :np.ndarray, y :np.ndarray) -> float:
        intensity_lambda = np.exp(x)
        array = np.empty(len(x))
        for i in range(len(x)):
            array[i]  = intensity_lambda[i] ** y[i] * np.exp(-intensity_lambda[i]) / np.math.factorial(y[i])
        return np.prod(array)
    

    def get_log_likelihood_y_given_x(self, x: np.ndarray, y: np.ndarray) -> float:
        intensity_lambda = np.exp(x)
        y = y.astype(float)
        #x_y = x * y
        #log_fac_y = np.log(float(np.math.factorial(y)))
        #log_likelihood = x_y - intensity_lambda - log_fac_y
        log_likelihood = [y_i * x_i - intensity_lambda_i - np.log(float(np.math.factorial(y_i))) for x_i, y_i, intensity_lambda_i in zip(x, y, intensity_lambda)]
        return np.sum(log_likelihood)



    
    @staticmethod
    def distance_matrix_one_dimension(vec_1, vec_2) -> np.ndarray:
        return distance_matrix(vec_1.reshape(-1,1), vec_2.reshape(-1,1))
    
    def make_covariance_matrix(self, S: np.ndarray, T = np.empty((1))) -> np.ndarray:
        
        start_t = time.time()
        sigma = self.model_parameters["sigma"]
        
        # Split the yx and z
        S_z = S[:,2]
        S_yx = S[:,0:2]

        # This function makes the covariance matrix for the model
        Dz_matrix = self.distance_matrix_one_dimension(S_z,S_z)
        Dyx_matrix = distance_matrix(S_yx,S_yx)
        Dt_matrix = self.distance_matrix_one_dimension(T,T)

        Sigma = self.covariance_function_z(Dz_matrix)
        Sigma = Sigma * self.covariance_function_yx(Dyx_matrix)
        Sigma = Sigma * self.covariance_function_temporal(Dt_matrix)
        Sigma = Sigma * sigma**2

        end_t = time.time()
        self.update_timing("make_covariance_matrix", end_t - start_t)
        return Sigma
      
    
    def make_covariance_matrix_2(self, S_1: np.ndarray, S_2: np.ndarray, T_1 = np.empty(1), T_2 = np.empty(1)) -> np.ndarray:
        start_t = time.time()
        # TODO: able to add just one time point
        sigma = self.model_parameters["sigma"]

        S_z_1 = S_1[:,2]
        S_yx_1 = S_1[:,0:2]
        S_z_2 = S_2[:,2]
        S_yx_2 = S_2[:,0:2]

        Dyx_matrix = distance_matrix(S_yx_1,S_yx_2)
        Dz_matrix = self.distance_matrix_one_dimension(S_z_1,S_z_2)
        Dt_matrix = self.distance_matrix_one_dimension(T_1,T_2) 
        Sigma = self.covariance_function_z(Dz_matrix)
        Sigma = Sigma * self.covariance_function_yx(Dyx_matrix)
        Sigma = Sigma * self.covariance_function_temporal(Dt_matrix)
        Sigma = Sigma * sigma**2 

        end_t = time.time()
        self.update_timing("make_covariance_matrix", end_t - start_t)
        return Sigma


    def predict(self, S_B: np.ndarray, T_B: np.ndarray):

        start_t = time.time()

        if self.print_while_running:
            print(time_now_str(), f"[INFO] [MODEL] Predicting values for {len(T_B)} points")

        if self.data_dict["has_data"] == False:
            if self.print_while_running:
                print(time_now_str() ,"[INFO] [MODEL] No data in model, predicting based on prior mean function")

            #mu_B = self.prior_mean_function(S_B, T_B) # REMOVE
            mu_B = self.prior.get_prior_S_T(S_B, T_B)
            Sigma_BB = self.make_covariance_matrix(S_B, T_B)
            return mu_B, Sigma_BB
        

        # Predict x_B based on x_A_est and Sigma_est
        # s_B: The points to predict
        # s_A: The points used for estimating x_A_est
        # x_A_est: The estimated x_A
        # Sigma_est: The estimated covariance matrix of x_A_est

        # Load data from memory
        S_A = self.data_dict["S"]
        T_A = self.data_dict["T"]
        x_A_est = self.data_dict["x"]
        Sigma_AA_est = self.data_dict["Sigma"]
        P = self.data_dict["P"]

        # Get the cross covariance matrix
        Sigma_AB = self.make_covariance_matrix_2(S_B, S_A, T_B, T_A)
        Sigma_BA = Sigma_AB.T

        # Get the covariance matrix of x_B
        Sigma_BB = self.make_covariance_matrix(S_B, T_B)
        # This should already be calculated
        Sigma_AA = self.make_covariance_matrix(S_A, T_A)

        # Get mean of x_B and x_A
        mu_A = self.data_dict["mu"]
        mu_B = self.prior.get_prior_S_T(S_B, T_B)    

        # check if Sigma_AA_inv is computed
        # Getting the invese of Sigma_AA is the second most expensive part of the algorithm
        if "Sigma_AA_inv" not in self.data_dict:
            self.data_dict["Sigma_AA_inv"] = np.linalg.inv(Sigma_AA) 
        else:
            # check if the dimension of Sigma_AA_inv fits with Sigma_AA
            # We do not compute the inverse of Sigma_AA if it is not needed
            if self.data_dict["Sigma_AA_inv"].shape[0] != Sigma_AA.shape[0]:
                # TODO: THere is some bug here, the woodsburry sherman inverse can be wrong: Answer, some points where extremely close to each other
                # That needs to be detected without computing the inverse of Sigma_AA the slow way
                k = Sigma_AA.shape[0] 
                n = k - self.data_dict["Sigma_AA_inv"].shape[0]
                m = k - n
         
                A_mat = Sigma_AA[:m,:m]
                B_mat = Sigma_AA[:m,m:]
                D_mat = Sigma_AA[m:,m:]
        
                self.data_dict["Sigma_AA_inv"] = self.inverse_matrix_block_symetric(A_mat, B_mat, D_mat, self.data_dict["Sigma_AA_inv"])

                    



        # Get conditional mean and covariance matrix
        Sigma_AAP_inv = self.data_dict["Sigma_P_inv"]   
        Sigma_AA_inv = self.data_dict["Sigma_AA_inv"]
        x_B_est = mu_B + Sigma_AB @ Sigma_AA_inv @ (x_A_est - mu_A)
        Sigma_BB_est = Sigma_BB - Sigma_AB @ Sigma_AAP_inv @ Sigma_BA


        end_t = time.time() 
        self.update_timing("predict", end_t - start_t)

        if self.print_while_running:
            print(time_now_str(), f"[INFO] [MODEL] Time for predicting {len(T_B)} points: {end_t - start_t:.2f} s")

        return x_B_est, Sigma_BB_est
    

    def estimate_parameters(self):
        # Estimate the parameters of the model
        # This is done by maximizing the likelihood of the model

        # Load data from memory
        S = self.all_data_dict["S"]
        T = self.all_data_dict["T"]
        Y = self.all_data_dict["y"]
        mu = self.all_data_dict["mu"]
        x = self.all_data_dict["x"]

        # Get the parameters of the model
        
        
        return None
    
    def reduce_points(self, S, T, Y):
        # This function reduces the points added to the model
        # Here we take the average position and time and add the Y values
        reduce_factor = self.model_parameters["reduce_factor"]
        n = len(Y)
        n_new = int(n * reduce_factor)
        

    
    
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
        x, Sigma, P, norm_list = self.fit(mu, mu, Y, Sigma)

        # Save the fitted data
        self.data_dict["x"] = x
        self.data_dict["P"] = P
        

        return x, Sigma

    def add_new_values(self, S_new: np.ndarray, T_new: np.array, Y_new: np.array):
        # Add new values to the model
        # S_new: new spatial points
        # Y_new: new observations
        # T_new: new temporal points


        start_t = time.time()           # Start the timer
        if self.print_while_running:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(now_str, f"[ACTION] Adding {len(T_new)} new values to the model")

        if len(Y_new) == 0:
            if self.print_warnings:
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(now_str, "[WARNING] [MODEL] No values to add to the model")
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
            P_old = self.data_dict["P"]


            # Get the dimensions of the old and new data
            n_old, n_new = len(y_old), len(Y_new)


            # Join the old and new data
            s = np.concatenate((s_old, S_new))
            t = np.concatenate((t_old, T_new))
            y = np.concatenate((y_old, Y_new))
            #mu_new = self.prior_mean_function(S_new, T_new) # REMOVE
            mu_new = self.prior.get_prior_S_T(S_new, T_new)
            mu = np.concatenate((mu_old, mu_new))

            # Get the cross covariance matrix
            # This can be done more effieciently
            Sigma = self.make_covariance_matrix(s, t)



            # Get the initial guess for x
            # this is based on the previouse estimate of x and the prior mean
            # Smarter way of doing this? # TODO:
            # Maybe use some simple extrapolation from the previouse fitted data
            
            # Use the five last values in the previouse estimate of x
            new_guess = np.repeat(np.mean(x_old[-5:]), n_new)
            new_guess = np.log(Y_new + 1) # This is a better guess

            #new_guess = 
            #x_init = np.concatenate((x_old, mu_new))
            x_init = np.concatenate((x_old, new_guess))

            # Fit the model 
            x, Sigma, P, norm_list = self.fit(x_init, mu, y, Sigma)

          
            # Save the data
            self.data_dict["has_data"] = True
            self.data_dict["S"] = s
            self.data_dict["T"] = t
            self.data_dict["y"] = y
            self.data_dict["mu"] = mu
            self.data_dict["Sigma"] = Sigma
            self.data_dict["x"] = x
            self.data_dict["P"] = P

            # Add the batch number
            batch_num = np.max(self.data_dict["batch"]) + 1
            self.data_dict["batch"] = np.concatenate((self.data_dict["batch"], np.repeat(batch_num, n_new)))
            


        # Update the all_data_dict
        self.update_all_data(n_new)
        
        end_t = time.time()             # End the timer
        # Update the timing data
        self.update_timing("add_values", end_t - start_t)

        
        




        


    def fit(self, x_init, mu, y, Sigma, max_iter: int = 40, tol: float = 1):
        # The model fiting is done by maximizing 
        # This becomes a version of the Newton-Raphson method

        # x_init: initial quess for the intensity
        # mu: prior mean
        # y: observations, these are observed at points S
        # Sigma: unconditional covariance matrix for the intensity

        if self.print_while_running:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            print(now_str, "[ACTION] [MODEL] Fitting model to new datapoints")

        start_t = time.time()
        
        if self.data_dict["has_data"] == False:
            n_old = 0
            n_new = len(y)
        else:
            n_old = self.data_dict["y"].shape[0]
            n_new = len(y) -  self.data_dict["y"].shape[0] 

        x = x_init                  # Initial guess
        norm_list = []              # List of change between x_i and x_i-1

        # Matices used in the fit loop
        P = 0
        Sigma_P_inv = 0
        Sigma_at_Sigma_P_inv = 0

        for i in range(max_iter):           
            x_prev = x
            bder = np.exp(x)
            b2der =  np.exp(x)
            z = (y - bder + x * b2der)  *  1 / b2der
            P = np.diag(1 / b2der)

            # This is probably the most expensive part of the algorithm
            Sigma_P_inv = np.linalg.inv(Sigma + P)
            Sigma_at_Sigma_P_inv = Sigma @ Sigma_P_inv
            x = mu + Sigma_at_Sigma_P_inv @ (z - mu)
            

            # Add the norm to the list 
            norm_list.append(np.linalg.norm(x - x_prev))   

            if np.isnan(P).any():
                if self.print_warnings:
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(now_str,'[WARNING] [MODEL] P contains nan values')
                break
            
            if n_old > 0 and False:
                print(i, np.linalg.norm(x_init[:n_old] - x[:n_old])) #REMOVE
                ids = np.arange(n_old + n_new)
                plt.plot(x_init[:n_old], label="x_init", c="red")
                plt.plot(ids[n_old:], x_init[n_old:], label="x_init", c="blue")
                plt.plot(x, label="x",c="green")
                plt.legend()
                plt.show()

                plt.plot(x_init, label="x_init")
                plt.plot(x, label="x")
                plt.legend()
                plt.show()
        
            if np.linalg.norm(x - x_prev) < tol:

                # Store the convergence data
                self.convergence_data["norm_list"].append(norm_list)
                self.convergence_data["iterations"].append(i)
                self.convergence_data["time"].append(time.time() - start_t)
                self.convergence_data["n_points"].append(len(y))


                if self.print_while_running:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    print(now_str, '[INFO] [MODEL] Fitting converged after {} iterations'.format(i + 1))
                break
        
        # Need to store the Sigma_P_inv for the predict function
        self.data_dict["Sigma_P_inv"] = Sigma_P_inv
        Sigma_est = Sigma - Sigma_at_Sigma_P_inv @ Sigma

        end_t = time.time() 
        self.update_timing("fit_model", end_t - start_t)

        if self.print_while_running:   
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            print(now_str, "[INFO] [MODEL] \t Norm list: ", [v.round(5) for v in norm_list])
            print(now_str, "[INFO] [MODEL] \t Time for fitting: ", round(end_t - start_t,3), " s")

        return x, Sigma_est,P, norm_list


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
        self.data_dict["P"] = old_data["P"][ind][:,ind]
        self.data_dict["batch"] = old_data["batch"][ind]
        self.data_dict["has_data"] = False # This will be true after the fit

        # Refit the model
        cov_mat = self.make_covariance_matrix(self.data_dict["S"], self.data_dict["T"])
        self.data_dict["x"], self.data_dict["Sigma"], self.data_dict["P"], norm_list = self.fit(self.data_dict["x"], self.data_dict["mu"], self.data_dict["y"], cov_mat)

        # Adding the unchanged data
        self.data_dict["has_data"] = True #old_data["has_data"]
        
        t_end = time.time()

        # Store timing
        self.update_timing("down_sample_points", t_end - t_start)
        if self.print_while_running:
            print(time_now_str(), "[ACTION] [MODEL] Down sampling points done")
            print(time_now_str(), "[TIMING] [MODEL] \t Time for downsampling: ", round(t_end - t_start,3), " s")


    def update_all_data(self, n_new: int):

        if self.all_data_dict["has_data"] == False:
            # Add the data to the all_auv_data

            # Vectors 
            self.all_data_dict["S"] = self.data_dict["S"]
            self.all_data_dict["T"] = self.data_dict["T"]
            self.all_data_dict["mu"] = self.data_dict["mu"]
            self.all_data_dict["x"] = self.data_dict["x"]
            self.all_data_dict["y"] = self.data_dict["y"]
            self.all_data_dict["batch"] = self.data_dict["batch"]
            
            # Diagonal of the matrices
            self.all_data_dict["dSigma"] = np.diag(self.data_dict["Sigma"])

            self.all_data_dict["has_data"] = True
        
        else:
             # Add the data to the all_auv_data

            # Vectors 
            self.all_data_dict["S"] = np.concatenate((self.all_data_dict["S"], self.data_dict["S"][-n_new:]))
            self.all_data_dict["T"] = np.concatenate((self.all_data_dict["T"], self.data_dict["T"][-n_new:]))
            self.all_data_dict["mu"] = np.concatenate((self.all_data_dict["mu"], self.data_dict["mu"][-n_new:]))
            self.all_data_dict["x"] = np.concatenate((self.all_data_dict["x"], self.data_dict["x"][-n_new:]))
            self.all_data_dict["y"] = np.concatenate((self.all_data_dict["y"], self.data_dict["y"][-n_new:]))
            self.all_data_dict["batch"] = np.concatenate((self.all_data_dict["batch"], self.data_dict["batch"][-n_new:]))

            # Diagonal of the matrices
            self.all_data_dict["dSigma"] = np.concatenate((self.all_data_dict["dSigma"], np.diag(self.data_dict["Sigma"])[-n_new:]))            


    def save(self, folder, name):
        # This function saves the model to a file
        # This file can be loaded later   

        if self.print_while_running:
            print(time_now_str(), "[ACTION] [MODEL] Saving model")
            print(time_now_str(), "[INFO] [MODEL] Saving model to: ", folder + "/model_data_"  + name + ".pickle")

        data = {
            "all_data": self.all_data_dict,
            "data_in_model": self.data_dict,
            "convergence_data": self.convergence_data,
            "timing_data": self.timing_data,
            "parameters": self.model_parameters
        }

        with open(folder + "/model_data_"  + name + ".pickle", 'wb') as f:
            pickle.dump(data, f)
        

    def load(self, folder, name):
        
        if self.print_while_running:
            print(time_now_str(), "[ACTION] [MODEL] Loading model")
            print(time_now_str(), "[INFO] [MODEL] Loading model from: ", folder + "/model_data_"  + name + ".pickle")

        # This function loads the model from a file
        with open(folder + "/model_data_"  + name + ".pickle", 'rb') as f:
            data = pickle.load(f)

        self.all_data_dict = data["all_data"]
        self.data_dict = data["data_in_model"]
        self.convergence_data = data["convergence_data"]
        self.timing_data = data["timing_data"]
        self.model_parameters = data["parameters"]


    def print_timing_data(self): 
        if self.timing:
            for key in self.timing_data.keys():
                print("Function: \t", key)
                timing_dat = self.timing_data[key]

                # Nicer formating
                print("\t total calls: ", round( timing_dat["counter"],2))
                print("\t Average time: ", round( timing_dat["total_time"] / timing_dat["counter"], 2), " s")
                print("\t Total time: ", round( timing_dat["total_time"], 2), " s")

    def print_data_shape(self):
        if self.data_dict["has_data"] == True:
            shape_data = []
            for key in self.data_dict.keys():
                row_dat = [key]
                if isinstance(self.data_dict[key], np.ndarray):
                    row_dat.append(str(self.data_dict[key].shape))
                else:
                    print("")
                    row_dat.append(str(type(self.data_dict[key])))
                shape_data.append(row_dat)
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(now_str, "[INFO] Data in model: ")
            for row in shape_data:
                print("{: >12} {: >12} ".format(*row))

    # Many get and set methods 

    def get_t_now(self):
        return self.data_dict["T"][-1]
    
    def get_s_now(self):
        return self.data_dict["S"][-1]

    def update_timing(self, func_name, t):
        if self.timing:
            if func_name in self.timing_data.keys():
                self.timing_data[func_name]["counter"] += 1 
                self.timing_data[func_name]["total_time"] += t
                self.timing_data[func_name]["time_list"].append(t)

            else:
                self.timing_data[func_name] = {}
                self.timing_data[func_name]["counter"] = 1 
                self.timing_data[func_name]["total_time"] = t
                self.timing_data[func_name]["time_list"] = [t]
    
    def inverse_matrix_block_symetric(self, A, B, D, A_inv):
        # inverting a matrix with the block shape 
        # | A   B | 
        # | B^T C |
        # where A^-1 is already calculated

        n = A.shape[0]
        m = D.shape[0]
        inverted_matrix = np.zeros((n+m,n+m))

        U = B.T @ A_inv
        V = U.T 

        S = np.linalg.inv(D - B.T @ A_inv @ B)
        
        V_at_S = V @ S

        inverted_matrix[0:n,0:n] = A_inv + V_at_S @ U
        inverted_matrix[n:(n+m),0:n] = - S @ U
        inverted_matrix[0:n,n:(n+m)] = - V_at_S
        inverted_matrix[(n):(n+m),(n):(n+m)] = S

        return inverted_matrix

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from Field import Field
    from Prior import Prior
    
    

    from plotting.ModelPlotting import ModelPlotting

    """
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
    """
    
    prior = Prior(parameters={"beta_0":0.5,
                               "beta_1":2,
                               "beta_2":1 / 6**2,
                               "peak_depth_min":10,
                               "peak_depth_max":60})
    
    # Test the model
    model = Model(prior = prior, print_while_running=True)


    time_1 = 24 * 3600 / 2

    # Get the field
    field = Field(print_while_running=True)
    # Generate the field
    n_s = [11,12,13] # 
    n_t = 14 # 
    S_lim = [[0,2000],[0,2000],[0,90]] # 2000 meters in the x and y direction, 50 meters in the z direction
    T_lim = [0,24 * 3600] # 24 hours
    field.generate_field_x(S_lim,T_lim,n_s,n_t)

    model_plotter = ModelPlotting(field, model)
    print("###############################################")
    print("#########   Test Random values ################")
    print("###############################################")

    model_plotter.plot_random_observations(field, model, n = 200)


    print("###############################################")
    print("#########   Test diving auv ################")
    print("###############################################")

    # Setting up a new model
    model = Model(prior = prior, print_while_running=True)
    model_plotter.plot_diving_AUV(field, model)


    print("###############################################")
    print("#########  Test adding consecutive values ####")
    print("###############################################")

    # Setting up a new model
    from PathPlanner import PathPlanner
    from Boundary import Boundary
    model = Model(prior = prior, print_while_running=False)
    boundary = Boundary("/src/csv/simulation_border_xy.csv", file_type="xy")
    path_planner = PathPlanner(boundary, sampling_frequency=0.1)
    model_plotter.plot_adding_consecutive_paths(field, model, path_planner, n = 200)
    model_plotter.plot_timing_data(model, "consecutive_paths")
  



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
    plt.savefig("figures/tests/Model_test/Model/Down_sampling" + ".png")
    plt.close()


    # Try to move in a yo-yo pattern
    start = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(10,40)])
    end = np.array([np.random.uniform(0,2000), np.random.uniform(0,2000), np.random.uniform(10,40)])
    yoyo_depth_limits = [end[2] + 10, end[2] - 10]
    path = path_planner.yo_yo_path(start, end, yoyo_depth_limits)

    S_path = path["S"]
    T_path = path["T"]
    observations = field.get_observation_Y(S_path, T_path)

    model = Model(prior=prior,print_while_running=True)
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
    plt.savefig("figures/tests/Model_test/Model/Yo_yo" + ".png")
    plt.close()

    # Save the model
    model.save("src/mission_data/test_store_model/", "test")

    model_before = model
    
    # Load the model
    model = Model(prior=prior, print_while_running=True)
    model.load("src/mission_data/test_store_model/", "test")



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
    plt.savefig("figures/tests/Model_test/Model/Yo_yo_load" + ".png")
    plt.close()











  

