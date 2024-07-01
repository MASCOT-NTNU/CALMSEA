import numpy as np

class Prior:
    def __init__(self, 
                 parameters = {
                        "beta_0": 0,              # The mean intensity
                        "beta_1": 3,              # The amplitude of the intensity
                        "beta_2": 2 / 6**2,              # related to the width of the intensity
                        "peak_depth_min": 15,           # The minimum depth of the peak
                        "peak_depth_max": 40            # The maximum depth of the peak
                 }):
        
        self.intensity_parameters = parameters


    def get_prior_S_T(self, S, T):
        peak_depth_max = self.intensity_parameters["peak_depth_max"]
        peak_depth_min = self.intensity_parameters["peak_depth_min"]
        beta_0 = self.intensity_parameters["beta_0"]
        beta_1 = self.intensity_parameters["beta_1"]
        beta_2 = self.intensity_parameters["beta_2"]

        # The intesity function is dependant on the depth and time of day 
        S_z = S[:,2] # The depth
        phase = 2 * 3.1415926 * T /(24 * 3600)
        peak_depth = (np.sin(phase) + 1) / 2  * (peak_depth_max - peak_depth_min) + peak_depth_min
        return np.exp(-(peak_depth - S_z)**2 * beta_2) * beta_1 + beta_0
    
    def get_peak_depth(self, T):
        peak_depth_max = self.intensity_parameters["peak_depth_max"]
        peak_depth_min = self.intensity_parameters["peak_depth_min"]
        phase = 2 * 3.1415926 * T /(24 * 3600)
        return (np.sin(phase) + 1) / 2 * (peak_depth_max - peak_depth_min) + peak_depth_min
    

    def set_parameter(self, parameter_name: str, value):
        if parameter_name not in self.intensity_parameters:
            print(f"[WARNING] [PRIOR] Parameter {parameter_name} not in the intensity parameters")
        self.intensity_parameters[parameter_name] = value

    
    def read_echo_sounder_csv(self, file_path):

        """
        This function reads the csv from the echo sounder 

        """






    
    
    



if __name__ == "__main__":

    prior = Prior()
    S = np.array([[1,2,3],[1,2,3],[1,2,3]])
    T = np.array([1,2,3])
    print(prior.get_prior_S_T(S,T))

    prior.set_parameter("beta_2", 0.058318251804938936)

    n = 100
    t = np.repeat(np.random.uniform(0, 24 * 3600),n)
    x = np.repeat(np.random.uniform(-200, 2200),n)
    y = np.repeat(np.random.uniform(-200, 2200),n)
    z = np.linspace(0,100, 100)

    S = np.array([x.T,y.T,z.T]).T
    T = np.array([t])

    print(S.shape)

    intensity = prior.get_prior_S_T(S,T)


    print(intensity.shape)

    # Get parameters 
    beta_0 = prior.intensity_parameters["beta_0"]
    beta_1 = prior.intensity_parameters["beta_1"]
    beta_2 = prior.intensity_parameters["beta_2"]
    peak_depth_min = prior.intensity_parameters["peak_depth_min"]
    peak_depth_max = prior.intensity_parameters["peak_depth_max"]

    width = np.sqrt(1/ beta_2)

    

    peak_depth = prior.get_peak_depth(t[0])
    import matplotlib.pyplot as plt
    plt.plot(z,intensity[0])
    plt.axvline(peak_depth, color="red", linestyle="--")
    plt.axvline(peak_depth + width, color="black", linestyle="--")
    plt.axvline(peak_depth - width, color="black", linestyle="--")
    plt.axhline(0, color="black")
    plt.show()


    # plot the peak depth
    t = np.linspace(0,24 * 3600, 100)
    peak_depth = prior.get_peak_depth(t)
    plt.plot(t, peak_depth)
    plt.show()
