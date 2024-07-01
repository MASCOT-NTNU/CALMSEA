import numpy as np
from scipy.stats import norm


from Field import Field

class FieldLogNormal(Field):

    def __init__(self, 
                 phi_spatial_z = 1/4,
                 phi_spatial_yx = 1 / 200,
                 phi_temporal = 1 / 7200,
                 sigma = 2,
                 tau = 1,
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
            "sigma": sigma,                      # The standard deviation of the field
            "tau": tau                          # The standard deviation of the observation noise
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


    def get_observation_Y(self, S: np.ndarray, T) -> np.ndarray:
        intensity_x = self.get_intensity_x(S,T)
        tau = self.covariance_parameters["tau"]
        log_Y = np.random.normal(intensity_x, tau)
        return np.exp(log_Y)
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time 
    from plotting.field_plotting import FieldPlotting

    tau = 2

    field = FieldLogNormal(tau=tau)
    print(field.intensity_parameters)
    print(field.covariance_parameters)
    print(field.field)

    S_lim = [[0,2000],[2000,4500],[0,100]] # 2000 meters in the x and y direction, 50 meters in the z direction
    T_lim = [0,24 * 3600] # 24 hours
    # Each dimension needs to have at least 4 points for the interpolation to use "cubic" method
    # Else it will use "linear" method
    n_s = [11,12,13] #  points in the x and y direction, 50 points in the z direction
    n_t = 20 # n_t temporal points
    t1 = time.time()
    field.generate_field_x(S_lim,T_lim,n_s,n_t)
    t2 = time.time()
    print(f"Time to generate field: {t2-t1:.2f} seconds")


    # Plot the field for fixed values of x and y and random values of z and t

    n = 100
    t = np.repeat(np.random.uniform(0,24 * 3600),n)
    Sx = np.repeat(np.random.uniform(0,2000),n)
    Sy = np.repeat(np.random.uniform(2000,4500),n)
    Sz = np.linspace(0,50,n)
    S = np.array([Sy,Sx,Sz]).T
    Y = field.get_observation_Y(S,t)
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(Sz,Y)
    ax[1].scatter(Sz,Y)
    plt.show()


    # Check if it is normal distributed
    fig, ax = plt.subplots(3,3, figsize=(15,15))
    # Title
    fig.suptitle('Testing that observations from single locations are normal distributeted', fontsize=16)
    n = 1000
    for i in range(3):
        for j in range(3):
            t = np.repeat(np.random.uniform(0,24 * 3600),n)
            Sy = np.repeat(np.random.uniform(0,2000),n)
            Sx = np.repeat(np.random.uniform(2000, 4500),n)
            Sz = np.repeat(np.random.uniform(0,50),n)

            S = np.array([Sy,Sx,Sz]).T
            Y = field.get_observation_Y(S,t)
            x = field.get_intensity_x(S,t)
            #print(x)
            x_min = np.min(np.log(Y))
            x_max = np.max(np.log(Y))
            g = np.linspace(x_min,x_max,100)
            tau = field.covariance_parameters["tau"]
            sns.histplot(np.log(Y),stat="density", ax=ax[i,j])
            ax[i,j].plot(g, norm.pdf(g,loc=x[0],scale=tau), c="r")
    plt.show()