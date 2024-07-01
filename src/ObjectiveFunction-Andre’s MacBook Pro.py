import numpy as np
from scipy.stats import norm
from help_func import time_now_str


class ObjectiveFunction:

    def __init__(self, name, print_while_running = False):
        self.name = name
        self.print_while_running = print_while_running

    @staticmethod
    def sliding_average(x, window_size):
        return np.convolve(x, np.ones(window_size), 'valid') / window_size

    def objective_function(self, model, design_data): 
        # The model is fitted to the observed data 
        # The design_data contains the different possible designs, with
        # the prediction using the model
        if self.name == "max_expected_improvement":
            return self.of_max_expected_improvement(model, design_data)

        elif self.name == "average_expected_improvement":
            return self.of_average_expected_improvement(model, design_data)
        
        elif self.name == "max_variance":
            return self.of_max_variance(model, design_data)
        
        elif self.name == "max_expected_intensity":
            return self.of_max_expected_intensity(model, design_data)
        
        elif self.name == "average_expected_intensity":
            return self.of_average_expected_intensity(model, design_data)
        
        elif self.name == "top_p_expected_improvement":
            return self.of_top_p_expected_improvement(model, design_data)

        elif self.name == "random":
            return self.of_random(model, design_data)
        
        else:
            raise ValueError(f"Objective function {self.name} not implemented")
        
    def print_start(self, design_data):
        print(time_now_str(), f"[INFO] [OBJECTIVE FUNCTION] Running {self.name}]")
        print(time_now_str(), f"[INFO] [OBJECTIVE FUNCTION] number of designs evaluated {len(design_data)}")
        
    def expected_improvement(self, mu, Sigma, x_max):
        sd = np.sqrt(np.diag(Sigma))
        return (mu - x_max) * (1 - norm.cdf((x_max - mu) / sd))  + sd * norm.pdf((x_max -mu) / sd)
    
    def best_value(self, model):
        if model.data_dict["has_data"] == False:
            return 0 
        else: 
            return  np.nanmax(model.data_dict["x"])
        
    def of_max_expected_improvement(self, model, design_data):

        if self.print_while_running:
            self.print_start(design_data)

        # Define the current best value 
        best_value = self.best_value(model)

        # Find the best value for each design
        design_id = 0
        best_design = None
        best_expected_improvement = -np.inf
        of_values = []

        for i, design in enumerate(design_data):

            x_design = design["x_pred"]
            Sigma_design = design["Sigma_pred"]

            values = {}

            # Compute the expected improvement
            expected_improvement = self.expected_improvement(x_design, Sigma_design, best_value)

            values["expected_improvement"] = expected_improvement

            max_expected_improvement = np.nanmax(expected_improvement)

            values["max_expected_improvement"] = max_expected_improvement

            if self.print_while_running:
                print(time_now_str(), f"EI for design {i}: {max_expected_improvement}")
            if max_expected_improvement > best_expected_improvement:
                best_expected_improvement = max_expected_improvement
                best_design = design
                design_id = i

            of_values.append(values)

        if self.print_while_running:
            print(time_now_str(), f"Best EI: {best_expected_improvement}")
        return best_design, design_id, of_values
    
    def of_top_p_expected_improvement(self, model, design_data):

        # Define the current best value 
        max_x = np.nanmax(model.data_dict["x"])

        if self.print_while_running:
            self.print_start(design_data)

        # Find the best value for each design
        design_id = 0
        best_design = None
        best_value = -np.inf
        of_values = []

        p = 0.1

        for i, design in enumerate(design_data):

            x_design = design["x_pred"]
            Sigma_design = design["Sigma_pred"]

            # Compute the expected improvement
            expected_improvement = self.expected_improvement(x_design, Sigma_design, max_x)
            k = int(p * len(expected_improvement))
            top_k_expected_improvement = np.sort(expected_improvement)[-k:]

            if self.print_while_running:
                print(time_now_str(), f"EI for design {i}: {top_k_expected_improvement}")

            if np.nanmax(top_k_expected_improvement) > best_value:
                best_value = np.nanmax(top_k_expected_improvement)
                best_design = design
                design_id = i

            # Store the values
            values = {}
            values["expected_improvement"] = expected_improvement
            values["top_k_expected_improvement"] = top_k_expected_improvement
            of_values.append(values)
            
        return best_design, design_id, of_values
    
    def of_average_expected_improvement(self, model, design_data):
        # Define the current best value 
        best_value = np.nanmax(model.data_dict["x"])

        if self.print_while_running:
            self.print_start(design_data)

        # Find the best value for each design
        best_design = None
        best_expected_improvement = -np.inf

        for design in design_data:

            x_design = design["x_pred"]
            Sigma_design = design["Sigma_pred"]

            # Compute the expected improvement
            expected_improvement = self.expected_improvement(x_design, Sigma_design, best_value)
            average_expected_improvement = np.average(expected_improvement)
            if average_expected_improvement > best_expected_improvement:
                best_expected_improvement = average_expected_improvement
                best_design = design
            
        return best_design
    
    def of_max_expected_intensity(self, model, design_data):
        # Find the design with the highest expected intensity
        best_design = None
        best_expected_intensity = -np.inf

        for design in design_data:
            x_design = design["x_pred"]
            expected_intensity = np.nanmax(x_design)
            if expected_intensity > best_expected_intensity:
                best_expected_intensity = expected_intensity
                best_design = design

        return best_design
    
    def of_average_expected_intensity(self, model, design_data):
        # Find the design with the highest expected intensity
        best_design = None
        best_expected_intensity = -np.inf

        for design in design_data:
            x_design = design["x_pred"]
            expected_intensity = np.average(x_design)
            if expected_intensity > best_expected_intensity:
                best_expected_intensity = expected_intensity
                best_design = design

        return best_design


    
    def of_max_variance(self, model, design_data):
        # Find the design with the highest variance
        best_design = None
        best_variance = -np.inf

        for design in design_data:
            Sigma_design = design["Sigma_pred"]
            max_variance = np.nanmax(np.diag(Sigma_design))
            if max_variance > best_variance:
                best_variance = max_variance
                best_design = design

        return best_design
 
    
    def of_random(self, model, design_data):
        if self.print_while_running:
            self.print_start(design_data)
        # Select a random design
        return np.random.choice(design_data)

        

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
    

if __name__ == "__main__":
    
    improvement_names = ["max_expected_improvement","top_p_expected_improvement", "average_expected_improvement", "max_variance", "max_expected_intensity", "average_expected_intensity", "random"]

    import matplotlib.pyplot as plt

    from Field import Field
    from Prior import Prior
    from Model import Model
    from PathPlanner import PathPlanner
    from Boundary import Boundary
    
    

    from plotting.ModelPlotting import ModelPlotting
    
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

    # Get the observation
    s_a = np.array([0,0,0])
    s_b = np.array([200,200,20])
    S = np.linspace(s_a,s_b,100)
    dist = np.linalg.norm(s_b - s_a)
    T = np.linspace(0, dist/ 1 , 100)
    Y = field.get_observation_Y(S,T)
    model.add_new_values(S,T,Y)

    # Possible paths 
    boundary = Boundary()
    path_planner = PathPlanner(boundary = boundary, sampling_frequency= 1,print_while_running=True)

    s_start = S[-1]
    t_start = T[-1]

    paths = []
    for i in range(10):
        path = path_planner.get_random_path(s_start, t_start, distance=100)
        paths.append(path)

    path_predicitons = []
    path_true_values = []
    for path in paths:
        x_pred, Sigma_pred = model.predict(path["S"], path["T"])
        path_predicitons.append({"x_pred":x_pred, "Sigma_pred":Sigma_pred})
        true_x = field.get_intensity_x(path["S"], path["T"])
        path_true_values.append(true_x)

    # Check if the 
    reduce_factors = [i for i in range(1,30)]
    score = [[] for _ in range(len(paths))]
    for reduce_factor in reduce_factors:
        objetive_function = ObjectiveFunction("max_expected_improvement", print_while_running=False)

        path_predicitons_reduced = []
        for path_prediciton in path_predicitons:

            indecies = np.array([i for i in range(len(path_prediciton["x_pred"])) if i % reduce_factor == 0])
           
            x_pred_reduced = path_prediciton["x_pred"][indecies]
            print(len(x_pred_reduced))
            Sigma_pred_reduced = path_prediciton["Sigma_pred"][indecies][:,indecies]
            path_predicitons_reduced.append({"x_pred":x_pred_reduced, "Sigma_pred":Sigma_pred_reduced})
        
        best_design, design_id, of_values = objetive_function.objective_function(model, path_predicitons_reduced)

        for i in range(len(paths)):
            score[i].append(of_values[i]["max_expected_improvement"])
    for s in score:
        plt.plot(reduce_factors, s)
    plt.show()





    for improvement_name in improvement_names:
        
        objetive_function = ObjectiveFunction(improvement_name, print_while_running=True)

        best_design, design_id, of_values = objetive_function.objective_function(model, path_predicitons)

        for i in range(len(paths)):
            plt.plot(paths[i]["T"], path_predicitons[i]["x_pred"], label=f"Design {i} prediction")
            plt.plot(paths[i]["T"], path_true_values[i], label=f"True x {i}")
            Sigma_pred_design = path_predicitons[i]["Sigma_pred"]
            plt.fill_between(paths[i]["T"], path_predicitons[i]["x_pred"] - 2 * np.sqrt(np.diag(Sigma_pred_design)), path_predicitons[i]["x_pred"] + 2 * np.sqrt(np.diag(Sigma_pred_design)), alpha=0.5)
            plt.plot(paths[i]["T"], of_values[i]["expected_improvement"] * 100, label=f"Expected improvement {i}")
            plt.legend()
            plt.show()





        


        





