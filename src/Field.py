import numpy as np
from scipy.spatial import distance_matrix
from scipy import interpolate
import datetime
import os
import pickle


class Field:

    """
    This class is used to simulate a field of intensity values.
    The intensity is dependant on the depth and time of day.
    The field is simulated using a Gaussian process with a squared exponential kernel.
    
    """

    def __init__(self, 
                 phi_spatial_z = 1/4,
                 phi_spatial_yx = 1 / 200,
                 phi_temporal = 1 / 7200,
                 sigma = 2,
                 Beta = (0.1, 3, 2 / 6**2),
                 s_lim = None,
                 t_lim = None,
                 n_s = [None, None, None],
                 n_t = None,
                 print_while_running = False,
                 generate_field_x = False) -> None:
        
        # Intensity parameters
        self.intensity_parameters = {
            "beta_0": Beta[0],              # The mean intensity
            "beta_1": Beta[1],              # The amplitude of the intensity
            "beta_2": Beta[2],              # related to the width of the intensity
            "peak_depth_min": 15,           # The minimum depth of the peak
            "peak_depth_max": 40            # The maximum depth of the peak
        }
        self.beta_0 = Beta[0] # The mean intensity
        self.beta_1 = Beta[1] # The amplitude of the intensity
        self.beta_2 = Beta[2] # related to the width of the intensity
        self.peak_depth_min = 15 # The minimum depth of the peak
        self.peak_depth_max = 40 # The maximum depth of the peak

        # Covariance parameters
        self.covariance_parameters = {
            "phi_temporal": phi_temporal,       # The temporal correlation parameter
            "phi_spatial_z": phi_spatial_z,     # The spatial correlation parameter in the z direction
            "phi_spatial_yx": phi_spatial_yx,   # The spatial correlation parameter in the xy-plane
            "sigma": sigma                      # The standard deviation of the field
        }
        self.phi_temporal = phi_temporal # The temporal correlation parameter
        self.phi_spatial_z = phi_spatial_z # The spatial correlation parameter in the z direction
        self.phi_spatial_yx = phi_spatial_yx # The spatial correlation parameter in the xy-pane
        self.sigma = sigma # The standard deviation of the field

        # Field limits 
        self.field = {
            "S_lim": s_lim, # [[sx_start, sx_end],[sy_start],[sz_end]] in meters. This is the limit of the field as a box
            "T_lim": t_lim, # [t_start, t_end], in seconds. This says when the simulation starts and ends
            "n_sy": n_s[0], # Number of spatial points in the y direction
            "n_sx": n_s[1], # Number of spatial points in the x direction
            "n_sz": n_s[2], # Number of spatial points in the z direction
            "n_t": n_t # Number of temporal points
        }
        self.S_lim = s_lim # [[sx_start, sx_end],[sy_start],[sz_end]] in meters. This is the limit of the field as a box
        self.T_lim = t_lim # [t_start, t_end], in seconds. This says when the simulation starts and ends
        self.n_sy = n_s[0] # Number of spatial points in the y direction
        self.n_sx = n_s[1] # Number of spatial points in the x direction
        self.n_sz = n_s[2] # Number of spatial points in the z direction
        self.n_t = n_t # Number of temporal points

        # Field 
        self.field_function = None
        self.data_cube = None # The field as a 4D array
        self.field_save_path = "data/simulated_field/"
        
        self.Sy_ax = None # The y-axis of the field
        self.Sx_ax = None # The x-axis of the field
        self.Sz_ax = None # The z-axis of the field
        self.T_ax = None # The time axis of the field
        self.S_field = None
        self.T_field = None
        self.x_field = None

        # Print while running
        self.print_while_running = print_while_running

        # Generate the field
        if generate_field_x:
            self.generate_field_x(s_lim, t_lim, n_s, n_t)





    @staticmethod
    def distance_matrix_one_dimension(vec_1, vec_2) -> np.ndarray:
        return distance_matrix(vec_1.reshape(-1,1), vec_2.reshape(-1,1))
    
    def get_intensity_mu_x(self,S: np.ndarray,T: np.ndarray) -> np.ndarray:
        
        # Load 
        peak_depth_max = self.intensity_parameters["peak_depth_max"]
        peak_depth_min = self.intensity_parameters["peak_depth_min"]
        beta_0 = self.intensity_parameters["beta_0"]
        beta_1 = self.intensity_parameters["beta_1"]
        beta_2 = self.intensity_parameters["beta_2"]

        # The intesity function is dependant on the depth and time of day 
        S_z = S[:,2] # The depth
        phase = 2 * 3.1415926 * T /(24 * 3600)
        peak_depth = np.sin(phase) * (peak_depth_max - peak_depth_min) + peak_depth_min
        return np.exp(-(peak_depth - S_z)**2 * beta_2) * beta_1 + beta_0
    
    def get_yx_correlation(self,h_yx: np.ndarray) -> np.ndarray:
        # This is the correlation in the yx-plane, or north-east plane
        phi_spatial_yx = self.covariance_parameters["phi_spatial_yx"]
        return np.exp(-h_yx * phi_spatial_yx) * (1 + h_yx * phi_spatial_yx)
    
    def get_z_correlation(self,h_z: np.ndarray) -> np.ndarray:
        # This is the corrolation in the z direction, or the depth direction
        phi_spatial_z = self.covariance_parameters["phi_spatial_z"]
        return np.exp(-h_z * phi_spatial_z) * (1 + h_z * phi_spatial_z)
    
    def get_temporal_correlation(self,h_t: np.ndarray) -> np.ndarray:
        # This is the temporal correlation
        phi_temporal = self.covariance_parameters["phi_temporal"]
        return np.exp(-h_t * phi_temporal)
    
    def get_covariance_matrix(self,S: np.ndarray,T: np.ndarray) -> np.ndarray:
        # S: array of spatial values 3D
        # T: array of temporal values 1D

        # Separate the yx-plane and the z-plane
        S_yx = S[:,:2]
        S_z = S[:,2]

        # Get the distance matrices
        h_yx = distance_matrix(S_yx, S_yx)
        h_z = self.distance_matrix_one_dimension(S_z,S_z)
        h_temporal = self.distance_matrix_one_dimension(T, T)

        # Calculate the covariance matrices
        c_spatial_yx = self.get_yx_correlation(h_yx) 
        c_spatial_z = self.get_z_correlation(h_z)
        c_temporal = self.get_temporal_correlation(h_temporal)
        sigma = self.covariance_parameters["sigma"]
        return c_spatial_z * c_spatial_yx * c_temporal * sigma**2
    
    def get_covariance_matrix_2(self, S_1: np.ndarray, S_2: np.ndarray, T_1: np.ndarray, T_2: np.ndarray) -> np.ndarray:

        # Separate the yx-plane and the z-plane for the two sets of points
        S_z_1 = S_1[:,2]
        S_yx_1 = S_1[:,0:2]
        S_z_2 = S_2[:,2]
        S_yx_2 = S_2[:,0:2]

        # Calculate the distance matrices
        h_yx = distance_matrix(S_yx_1,S_yx_2)
        h_z = self.distance_matrix_one_dimension(S_z_1,S_z_2)
        h_temporal = self.distance_matrix_one_dimension(T_1,T_2) 

        # Calculate the covariance matrices
        c_spatial_yx = self.get_yx_correlation(h_yx) 
        c_spatial_z = self.get_z_correlation(h_z)
        c_temporal = self.get_temporal_correlation(h_temporal)

        # Return the covariance matrix
        return c_spatial_z * c_spatial_yx * c_temporal * self.sigma**2
    
    def sample_multivariate_normal(self,mu: np.ndarray, Sigma: np.ndarray, diag_v = 1e-10) -> np.ndarray:
        # return np.random.multivariate_normal(mu,Sigma)
        # This method is far less robust than the one above, but it is much faster
        diag = np.eye(len(mu)) * diag_v
        try:
            L = np.linalg.cholesky(Sigma + diag)
        except:
            print("Sigma is not positive definite", diag_v)
            print(np.min(Sigma))

            return self.sample_multivariate_normal(mu, Sigma, diag_v * 10)
        z = np.random.normal(0,1,len(mu))
        return mu + L @ z


    
    def simulate_intensity_x(self,S,T):
        mu = self.get_intensity_mu_x(S,T)
        Sigma = self.get_covariance_matrix(S,T)
        return self.sample_multivariate_normal(mu,Sigma)
    
    def simulate_intensity_x2(self,S,T):

        # Simulate the intensity field in several steps
        # 1. Simulate the intensity field at the first time step
        # 2. Simulate the intensity field at the second time step given the first time step
        # 3. Simulate the intensity field at the third time step given the first and second time step
        # 4. Continue until the last time step

        # 1. Simulate the intensity field at the first time step
        n_sx = self.field["n_sx"]
        n_sy = self.field["n_sy"]
        n_sz = self.field["n_sz"]
        n_spatial_points = n_sx * n_sy * n_sz
        mu_1 = self.get_intensity_mu_x(S[:n_spatial_points],T[:n_spatial_points])
        Sigma_1 = self.get_covariance_matrix(S[:n_spatial_points],T[:n_spatial_points])
        x_1 = self.sample_multivariate_normal(mu_1,Sigma_1)

        x = np.zeros_like(T)
        x[:n_spatial_points] = x_1

        mu_km1 = mu_1
        Sigma_km1 = Sigma_1
        if self.print_while_running:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(now_str, "[INFO] [FIELD] Simulating the intensity field")
            n_t = self.field["n_t"]
            print(f"######## n_t={n_t} ", end=" ")
        for k in range(1,self.field["n_t"]):
            if self.print_while_running:
                print(k, end=" ")
            S_k = S[k*n_spatial_points:(k+1)*n_spatial_points]
            T_k = T[k*n_spatial_points:(k+1)*n_spatial_points]
            S_km1 = S[(k-1)*n_spatial_points:k*n_spatial_points]
            T_km1 = T[(k-1)*n_spatial_points:k*n_spatial_points]
            x_km1, mu_km1, Sigma_km1 = self.simulate_xk_given_xkm1(S_k, T_k, S_km1, T_km1, mu_km1, Sigma_km1, x_1)
            x[k*n_spatial_points:(k+1)*n_spatial_points] = x_km1
        return x
    
    def simulate_xk_given_xkm1(self, S_k, T_k, S_km1, T_km1,mu_km1, Sigma_km1, x_km1):
        # Simulate x_k given x_km1
        # This is done by conditioning on the previous temporal values
        # This is done by conditioning on the previous temporal values
        # This will be "almost" correct. Because of the temoporal correlation being exponential
        # Then this will be correct. 
        mu_k = self.get_intensity_mu_x(S_k,T_k)
        Sigma_k = self.get_covariance_matrix(S_k,T_k)
        Sigma_k_km1 = self.get_covariance_matrix_2(S_k,S_km1,T_k,T_km1)
        
        Sigma_km1_inv = np.linalg.inv(Sigma_km1 + np.eye(len(T_km1)) * 1e-12)
        S_k_km1_at_S_km1_inv = Sigma_k_km1 @ Sigma_km1_inv
        mu_k_given_km1 = mu_k + S_k_km1_at_S_km1_inv  @ (x_km1 - mu_km1)
        Sigma_k_given_km1 = Sigma_k - S_k_km1_at_S_km1_inv  @ Sigma_k_km1.T
        return self.sample_multivariate_normal(mu_k_given_km1,Sigma_k_given_km1), mu_k, Sigma_k
        

        # generate a sample from the multivariate normal distribution
    
    def get_intensity_x(self,S,T):
        return self.interpolate_field(S,T)  
        
    
    def get_mean_lambda(self,S,T):
        return np.exp(self.get_intensity_x(S,T))
    
    def get_observation_Y(self, S :np.ndarray,T) -> np.ndarray:
        mean_lambda = self.get_mean_lambda(S,T)   
        # Some times there is an issue with mean_lambda
        
        try:
            return np.random.poisson(mean_lambda)
        except:
            print("[WARNING] Error in get_observation_Y")
            print("mean_lambda", mean_lambda)
            for s in S:
                if s < self.S_lim[0] or s > self.S_lim[1]:
                    print("s is outside the limits of the field")
            for t in T:
                if t < self.T_lim[0] or t > self.T_lim[1]:
                    print("t is outside the limits of the field")
                    
            print("S", S)
            print("T", T)
            return np.zeros_like(mean_lambda)

    
    
    def get_mean_lambda_from_x(self,x) -> np.ndarray:
        return np.exp(x)


    def generate_field2_x(self, S_lim, T_lim, n_s, n_t):
        ### THIS METHOD IS REDUNDANT AND SLOW: REMOVE
        # Generate the grid for the field
        S_y = np.linspace(S_lim[0][0],S_lim[0][1], n_s[0])
        S_x = np.linspace(S_lim[1][0],S_lim[1][1], n_s[1])
        S_z = np.linspace(S_lim[2][0],S_lim[2][1], n_s[2])          
        T = np.linspace(T_lim[0],T_lim[1],n_t)

        print(T_lim)
        # Store the axes
        self.Sy_ax = S_y
        self.Sx_ax = S_x
        self.Sz_ax = S_z
        self.T_ax = T

        S_yy, S_xx, S_zz, TT = np.meshgrid(S_y,S_x,S_z,T)
        S = np.array([S_yy.flatten(),S_xx.flatten(),S_zz.flatten()]).T
        T = TT.flatten()
        x = self.simulate_intensity_x(S[:,:3],T)    
        self.data_cube = np.reshape(x, (n_s[0],n_s[1],n_s[2],n_t))
    

        # Store the field
        self.S_lim = S_lim
        self.T_lim = T_lim
        self.n_s = n_s
        self.n_t = n_t
        self.S_field = S
        self.T_field = T
        self.x_field = x

        # Store the field function
        if min(n_s) > 4 and n_t > 4:
            self.field_function = interpolate.RegularGridInterpolator((S_y,S_x,S_z,self.T_ax),self.data_cube,bounds_error=False,fill_value=0)
            #self.field_function = interpolate.RegularGridInterpolator((S_y,S_x,S_z,self.T_ax),self.data_cube,bounds_error=False,fill_value=0, method="cubic")
        else:  
            self.field_function = interpolate.RegularGridInterpolator((S_y,S_x,S_z,self.T_ax),self.data_cube,bounds_error=False,fill_value=0, method="linear")
        return x
    
    def generate_field_x(self, S_lim, T_lim, n_s, n_t):

        # Generate the grid for the field
        Sy_ax = np.linspace(S_lim[0][0],S_lim[0][1], n_s[0])
        Sx_ax = np.linspace(S_lim[1][0],S_lim[1][1], n_s[1])
        Sz_ax = np.linspace(S_lim[2][0],S_lim[2][1], n_s[2])
        T_ax = np.linspace(T_lim[0],T_lim[1],n_t)

        if self.print_while_running:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(now_str, "[INFO] [FIELD] Generating the field")
            print(f"n_y: {n_s[0]}, n_x: {n_s[1]}, n_z: {n_s[2]}, n_t: {n_t}")
            print(f"A total of {n_s[0] * n_s[1] * n_s[2]} points in space, and {n_s[0] * n_s[1] * n_s[2] * n_t} in total.")

        # Store the axes
        #self.Sy_ax = Sy_ax # REMOVE
        #self.Sx_ax = Sx_ax # REMOVE
        #self.Sz_ax = Sz_ax # REMOVE
        #self.T_ax = T_ax   # REMOVE
        self.field["Sy_ax"] = Sy_ax
        self.field["Sx_ax"] = Sx_ax
        self.field["Sz_ax"] = Sz_ax
        self.field["T_ax"] = T_ax

        # Adding the number of spatial points and the number of temporal points
        #self.n_sy = n_s[0] # REMOVE
        #self.n_sx = n_s[1] # REMOVE
        #self.n_sz = n_s[2] # REMOVE
        #self.n_t = n_t # REMOVE
        self.field["n_sy"] = n_s[0]
        self.field["n_sx"] = n_s[1]
        self.field["n_sz"] = n_s[2]
        self.field["n_t"] = n_t


    
        S_yy, S_xx, S_zz, TT = np.meshgrid(Sy_ax,Sx_ax,Sz_ax,T_ax)
        S = np.array([S_yy.flatten(),S_xx.flatten(),S_zz.flatten()]).T
        T = TT.flatten()
        x = self.simulate_intensity_x2(S[:,:3],TT.flatten())

        self.data_cube = np.reshape(x, (n_s[0],n_s[1],n_s[2],n_t))
        self.field["data_cube"] = self.data_cube

        # Store the field
        #self.S_lim = S_lim # REMOVE
        #self.T_lim = T_lim # REMOVE
        #elf.n_s = n_s # REMOVE
        #self.n_t = n_t # REMOVE
        #self.S_field = S # REMOVE
        #self.T_field = T # REMOVE
        #self.x_field = x # REMOVE
        self.field["S_lim"] = S_lim
        self.field["T_lim"] = T_lim
        self.field["S_field"] = S
        self.field["T_field"] = T
        self.field["x_field"] = x
        self.field["n_s"] = n_s

        # Store the field function
        if min(n_s) > 4 and n_t > 4:
            self.field_function = interpolate.RegularGridInterpolator((Sy_ax,Sx_ax,Sz_ax,T_ax),self.data_cube,bounds_error=False,fill_value=0)
            #self.field_function = interpolate.RegularGridInterpolator((Sy_ax,Sx_ax,Sz_ax,T_ax),self.data_cube,bounds_error=False,fill_value=0, method='cubic')
        else:  
            self.field_function = interpolate.RegularGridInterpolator((Sy_ax,Sx_ax,Sz_ax,T_ax),self.data_cube,bounds_error=False,fill_value=0, method="linear")
        return x
    
    def interpolate_field_CN_s_t(self, s_new, t_new):

        # This is a closest neighbour interpolation
        # TODO: This can be vectorized, and faster, but there might not be a great need. 

        # Check if the field has been generated
        if self.field_function is None:
            print("Field has not been generated")
            return
        
        S_lim = self.field["S_lim"]
        # Check if the new values of S and T are within the limits of the field
        if s_new[0] < S_lim[0][0] or s_new[0] > S_lim[0][1]:
            print("S_new[0] is outside the limits of the field")
            return
        if s_new[1] < S_lim[1][0] or s_new[1] > S_lim[1][1]:
            print("S_new[1] is outside the limits of the field")
            return
        if s_new[2] < S_lim[2][0] or s_new[2] > S_lim[2][1]:
            print("S_new[2] is outside the limits of the field")
            return 
        if t_new < T_lim[0] or t_new > T_lim[1]:
            print("T_new is outside the limits of the field")
            return 

        Sy_ax = self.field["Sy_ax"]
        Sx_ax = self.field["Sx_ax"]
        Sz_ax = self.field["Sz_ax"]
        T_ax = self.field["T_ax"]
        data_cube = self.field["data_cube"]
        S_closest = [np.argmin(np.abs(Sy_ax - s_new[0])),np.argmin(np.abs(Sx_ax - s_new[1])),np.argmin(np.abs(Sz_ax - s_new[2]))]
        T_closest = np.argmin(np.abs(T_ax - t_new))
        
        return data_cube[S_closest[0],S_closest[1],S_closest[2],T_closest]
            
    
    def interpolate_field_CN(self, S, T):
        # This fuction interpolates the field using the closest neighbour
        n = len(T)
        x = np.zeros(n)
        for i in range(n):
            x[i] = self.interpolate_field_CN_s_t(S[i],T[i])
        return x
    
    def interpolate_field(self, S, T):
        S_join = np.column_stack((S,T))
        return self.field_function(S_join)
    
    def save_field(self, path, filename):
        # check if the field has been generated
        if self.field_function is None:
            print("Field has not been generated")
            return 
        
        store_dict = {
            "prior_intensity_parameters": self.intensity_parameters,
            "covariance_parmeters": self.covariance_parameters,
            "field": self.field
        }

        # Save the field
        file_path_name = path +"/field_"+ filename + ".pkl"
        with open(file_path_name, "wb") as f:
            pickle.dump(store_dict, f)

    def load_field_from_file(self, file_path):
        with open(file_path, "rb") as f:
            store_dict = pickle.load(f)

        if self.print_while_running:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(now_str, "[INFO] [FIELD] Loading the field from file", file_path)


        self.intensity_parameters = store_dict["prior_intensity_parameters"]
        self.covariance_parameters = store_dict["covariance_parmeters"]
        self.field = store_dict["field"]

        # Generate the interpolation function form the field
        Sy_ax = self.field["Sy_ax"]
        Sx_ax = self.field["Sx_ax"]
        Sz_ax = self.field["Sz_ax"]
        T_ax = self.field["T_ax"]
        data_cube = self.field["data_cube"]
        # Store the field function
        if min(n_s) > 4 and n_t > 4:
            self.field_function = interpolate.RegularGridInterpolator((Sy_ax,Sx_ax,Sz_ax,T_ax),data_cube,bounds_error=False,fill_value=0)
            #self.field_function = interpolate.RegularGridInterpolator((Sy_ax,Sx_ax,Sz_ax,T_ax),data_cube,bounds_error=False,fill_value=0, k=3)
        else:  
            self.field_function = interpolate.RegularGridInterpolator((Sy_ax,Sx_ax,Sz_ax,T_ax),data_cube,bounds_error=False,fill_value=0, method="linear")

    





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time 
    from plotting.field_plotting import FieldPlotting

    sigma = 2

    field = Field(sigma=sigma, print_while_running=True)
    field_plotting = FieldPlotting(field)

    S_lim = [[0,2000],[2000,4500],[0,50]] # 2000 meters in the x and y direction, 50 meters in the z direction
    T_lim = [0,24 * 3600] # 24 hours
    # Each dimension needs to have at least 4 points for the interpolation to use "cubic" method
    # Else it will use "linear" method
    n_s = [11,12,13] #  points in the x and y direction, 50 points in the z direction
    n_t = 20 # n_t temporal points
    t1 = time.time()
    
    field.generate_field_x(S_lim,T_lim,n_s,n_t)
    t2 = time.time()
    print(f"Time to generate field: {t2-t1:.2f} seconds")

    # Testing the prior intensity function
    n , m = 10, 100
    field_plotting.plot_prior_intensity(field, n,m)

    field_plotting.plot_lambda_slices(field)



    

    # Plot the field for fixed values of x and y and random values of z and t


    Sy = np.repeat(np.random.uniform(S_lim[0][0], S_lim[0][1]),10000)
    Sx = np.repeat(np.random.uniform(S_lim[1][0], S_lim[1][1]),10000)
    Sz = np.random.uniform(0,50,10000)
    S = np.array([Sy,Sx,Sz]).T
    T_random = np.random.uniform(0,24 * 3600,10000)
    x_random1 = field.interpolate_field_CN(S,T_random)
    x_random2 = field.interpolate_field(S,T_random)
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(Sz,T_random,c=x_random1)
    ax[1].scatter(Sz,T_random,c=x_random2)
    fig.savefig("figures/tests/Field/slice_zt.png")
    plt.close()

    
    field_plotting.plot_6_slices(field)
    field_plotting.plot_6_slices2(field)


    # Show how the different correlation parameters affect the field
    field_plotting.plot_covariance_function_illustration(field)


   


 



    # Testing how different correlation parameters look like
    phi_spatial_z_range = np.linspace(0.1,1,5)
    phi_spatial_yx_range = np.linspace(1/400,1/59,5)
    phi_temporal_range = np.linspace(1/7200,1/1000,5)

    field_plotting.plot_different_corr_parameters(field, phi_spatial_yx_range, phi_spatial_z_range, phi_temporal_range)
    
    


    # Save the field
    field.save_field("data/simulated_field/","test_field")

    # Load the field
    field2 = Field(print_while_running=True)
    field2.load_field_from_file("data/simulated_field/field_test_field.pkl")

    # Check if they are the same by drawing random samples 
    Sy = np.random.uniform(S_lim[0][0], S_lim[0][1],10000)
    Sx = np.random.uniform(S_lim[1][0], S_lim[1][1],10000)
    Sz = np.random.uniform(S_lim[2][0], S_lim[2][1],10000)
    S = np.array([Sy,Sx,Sz]).T
    T_random = np.random.uniform(T_lim[0],T_lim[1],10000)
    x_random1 = field.interpolate_field_CN(S,T_random)
    x_random2 = field2.interpolate_field_CN(S,T_random)
    print("Difference between the two fields CN", np.sum(np.abs(x_random1 - x_random2)))
    x2_ramdon1 = field.interpolate_field(S,T_random)
    x2_ramdon2 = field2.interpolate_field(S,T_random)
    print("Difference between the two fields interp",np.sum(np.abs(x2_ramdon1 - x2_ramdon2)))





  