import numpy as np
from scipy.stats import norm


class ObjectiveFunction:

    def __init__(self, name):
        self.name = name

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

        elif self.name == "random":
            return self.of_random(model, design_data)
        
        else:
            raise ValueError(f"Objective function {self.name} not implemented")
        
    def expected_improvement(self, mu, Sigma, x_max):
        sd = np.sqrt(np.diag(Sigma))
        return (mu - x_max) * (1 - norm.cdf((x_max - mu) / sd))  + sd * norm.pdf((x_max -mu) / sd)
        
    def of_max_expected_improvement(self, model, design_data):

        # Define the current best value 
        best_value = np.nanmax(model.data_dict["x"])

        # Find the best value for each design
        design_id = 0
        best_design = None
        best_expected_improvement = -np.inf

        for i, design in enumerate(design_data):

            x_design = design["x_pred"]
            Sigma_design = design["Sigma_pred"]

            # Compute the expected improvement
            expected_improvement = self.expected_improvement(x_design, Sigma_design, best_value)
            max_expected_improvement = np.nanmax(expected_improvement)
            if max_expected_improvement > best_expected_improvement:
                best_expected_improvement = max_expected_improvement
                best_design = design
                design_id = i
            
        return best_design, design_id
    
    def of_average_expected_improvement(self, model, design_data):
        # Define the current best value 
        best_value = np.nanmax(model.data_dict["x"])

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