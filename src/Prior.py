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
        peak_depth = np.sin(phase) * (peak_depth_max - peak_depth_min) + peak_depth_min
        return np.exp(-(peak_depth - S_z)**2 * beta_2) * beta_1 + beta_0
    

    def set_parameter(self, parameter_name: str, value):
        if parameter_name not in self.intensity_parameters:
            print(f"[WARNING] [PRIOR] Parameter {parameter_name} not in the intensity parameters")
        self.intensity_parameters[parameter_name] = value

    
    
    



if __name__ == "__main__":

    prior = Prior()
    S = np.array([[1,2,3],[1,2,3],[1,2,3]])
    T = np.array([1,2,3])
    print(prior.get_prior(S,T))
