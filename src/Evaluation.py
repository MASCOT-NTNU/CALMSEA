import numpy as np
import time 
import pickle

class Evaluation:

    def __init__(self, model, field):
        self.model = model
        self.field = field

        self.evaluation_points = None
        self.predicted_values = {}
        self.true_values = None
        self.store_path = None

        self.time_step = []
        self.evaluation = {
            "mse": [],
            "rmse": [],
            "dist_to_max": [],
            "dist_pred_max_to_max": [],
            "mse_lambda": []
        }
        self.evaluation_time = []
        

    def evaluate(self):
        t_start = time.time()

        # Generate the evaluation points
        s = np.linspace(0, 50, 100)
        t = np.repeat(self.model.get_t_now(), 100)
        self.evaluation_points = {"s": s, "t": t}

        # Use the model to predict the intensity at the evaluation points
        self.predicted_values["x"], self.predicted_values["Sigma"] = self.model.predict(s,t)

        # Get the true values at the evaluation points
        self.true_values = self.field.get_intensity_x(s, t)

        # Calculate the evaluation metrics
        self.evaluation["mse"].append(self.mse())
        self.evaluation["rmse"].append(self.rmse())
        self.evaluation["dist_to_max"].append(self.dist_to_max())
        self.evaluation["dist_pred_max_to_max"].append(self.dist_pred_max_to_max())
        self.evaluation["mse_lambda"].append(self.mse_lambda())
        self.time_step.append(self.model.get_t_now())
        t_end = time.time()
        self.evaluation_time.append(t_end - t_start)



    def update_model(self, model):
        self.model = model
        

    def mse(self):
        return np.mean((self.true_values - self.predicted_values["x"])**2)
    
    def rmse(self):
        return np.sqrt(self.mse())
    
    def dist_to_max(self):
        # The distance between the agent and the maximum value of the intensity
        s_now = self.model.get_s_now()
        s_max_ind = np.argmax(self.true_values)
        s_max = self.evaluation_points["s"][s_max_ind]
        return np.abs(s_now - s_max)
    
    def dist_pred_max_to_max(self):
        # The distance between the predicted maximum and the true maximum
        s_pred_max_ind = np.argmax(self.predicted_values["x"])
        s_pred_max = self.evaluation_points["s"][s_pred_max_ind]
        s_true_max_ind = np.argmax(self.true_values)
        s_true_max = self.evaluation_points["s"][s_true_max_ind]
        return np.abs(s_pred_max - s_true_max)
    
    def mse_lambda(self):
        # The mean squared error of the lambda parameter
        pred_lambda = np.exp(self.predicted_values["x"])
        true_lambda = np.exp(self.true_values)
        return np.mean((pred_lambda - true_lambda)**2)
    

    def save_evaluation(self, folder_path, name):
        # Save the evaluation metrics

        # The metrics being stored 
        store_dict = {
            "time_step": self.time_step,
            "evaluation_time": self.evaluation_time
        }
        for key in self.evaluation.keys():
            store_dict[key] = self.evaluation[key]

        with open(folder_path + "/evaluation_" + name + ".pkl", "wb") as f:
            pickle.dump(store_dict, f)

    def print_current_evaluation(self):
        # Print the current evaluation metrics
        print("Evaluation metrics:")
        for key in self.evaluation.keys():
            print(key, round(self.evaluation[key][-1],5))
        print("")



if __name__ == "__main__":
    # Import the model
    from Field import Field
    from Model import Model
    from ObjectiveFunction import ObjectiveFunction
    pass