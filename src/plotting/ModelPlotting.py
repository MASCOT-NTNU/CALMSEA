import matplotlib.pyplot as plt
import numpy as np
import time


class ModelPlotting:
    
    def __init__(self, model, field):
        self.model = model
        self.field = field

    @staticmethod
    def plot_random_observations(field, model, n = 200):
        "Add random observations to the model"
        Sx = np.random.uniform(0, 2000, n)
        Sy = np.random.uniform(0, 2000, n)
        Sz = np.random.uniform(0, 90, n)
        T = np.random.uniform(0, 24*3600, n)
        S = np.array([Sx, Sy, Sz]).T
        Y = field.get_observation_Y(S, T)
        model.add_new_values(S, T, Y)

        fitted_intensity = model.data_dict["x"][-n:]

        true_intensity = field.get_intensity_x(S, T)
        
        # Plot the true intensity vs the observed intensity
        plt.figure()
        plt.scatter(true_intensity, fitted_intensity)
        # Add a line with slope 1
        plt.plot([0, np.max(true_intensity)], [0, np.max(true_intensity)], color="red")

        plt.xlabel("True intensity (x)")
        plt.ylabel("Fitted intensity (x)")
        plt.title("True intensity vs fitted intensity for random observations")
        plt.savefig("figures/tests/Model/true_vs_fitted_intensity_random.png")
        plt.close()
        

        plt.figure()
        plt.scatter(np.exp(true_intensity), np.exp(fitted_intensity))
        # Add a line with slope 1
        plt.plot([0, np.max(np.exp(true_intensity))], [0, np.max(np.exp(true_intensity))], color="red")
        plt.xlabel("True intensity (lambda)")
        plt.ylabel("Fitted intensity (lambda)")
        plt.title("True intensity vs fitted intensity for random observations")
        plt.savefig("figures/tests/Model/true_vs_fitted_intensity_random_lambda.png")
        plt.close()




    def plot_diving_AUV(self, field, model):
        "Plotting a diving AUV in the field"
        n = np.random.randint(95, 105)
        m = np.random.randint(95, 105)
        
        # Get the points in the field
        Sz1 = np.linspace(0, 25 , n)
        x_p = np.random.uniform(0, 2000)
        y_p = np.random.uniform(0, 2000)
        Sx1 = np.repeat(x_p, n)
        Sy1 = np.repeat(y_p, n)
        S1 = np.array([Sx1, Sy1, Sz1]).T
        T1 = np.linspace(2000, 2200, n)

        # Get the values from the field 
        Y1 = field.get_observation_Y(S1, T1)
        x1_true = field.get_intensity_x(S1, T1)
        lambda1_true = np.exp(x1_true)

        # Predict the intensity at the points
        x1_pred, Sigma1_pred = model.predict(S1, T1)
        lambda1_pred = np.exp(x1_pred)
        

        # Add the values to the model
        model.add_new_values(S1, T1, Y1)

        # Get the fitted intensity
        x1_fit = model.data_dict["x"]
        x1_prior = model.data_dict["x"]
        Sigma1_fit = model.data_dict["Sigma"]
        lambda1_fit = np.exp(x1_fit)


        # Get the next points of the field 
        Sx2 = np.repeat(x_p, m)
        Sy2 = np.repeat(y_p, m)
        Sz2 = np.linspace(25, 50, m)
        S2 = np.array([Sx2, Sy2, Sz2]).T
        T2 = np.linspace(2200, 2400, m)

        # Precict the intensity at the next points
        x2_pred, Sigma2_pred = model.predict(S2, T2)
        

        # Get the values from the field
        Y2 = field.get_observation_Y(S2, T2)
        x2_true = field.get_intensity_x(S2, T2)
        lambda2_true = np.exp(x2_true)
        

        # Add the values to the model
        model.add_new_values(S2, T2, Y2)

        # Get the fitted intensity
        xall_fit = model.data_dict["x"]
        xall_prior = model.data_dict["x"]
        Sigmaall_fit = model.data_dict["Sigma"]
        lambdaall_fit = np.exp(xall_fit)

        # Combine the values
        Sall = np.concatenate([S1, S2])
        Szall = np.concatenate([Sz1, Sz2])
        Tall = np.concatenate([T1, T2])
        lambdaall_true = np.concatenate([lambda1_true, lambda2_true])

        # Plot the diving AUV
        fig, ax = plt.subplots(2, 3, figsize=(15,20))

        # Plot the first part of the diving AUV
        ax[0,0].plot(Sz1, lambda1_true, label="True intensity")
        ax[0,0].plot(Sz1, lambda1_pred, label="Predicted intensity")
        ax[0,0].scatter(Sz1, Y1, label="Observed intensity", color="red", alpha=0.5, marker="x")

        ax[1,0].plot(Sz1, x1_true, label="True intensity")
        ax[1,0].plot(Sz1, x1_pred, label="Predicted intensity")
        ax[1,0].plot(Sx1, x1_prior, label="Prior intensity")
        ax[1,0].fill_between(Sz1, x1_pred - 2 * np.sqrt(np.diag(Sigma1_pred)), x1_pred + 2 * np.sqrt(np.diag(Sigma1_pred)), alpha=0.2)
        ax[1,0].scatter(Sz1, np.log(Y1 +1), label="Log observarions", color="red", alpha=0.5, marker="x")

        # Plot the second part of the diving AUV 
        ax[0,1].plot(Sz1, lambda1_true, label="True intensity")
        ax[0,1].plot(Sz1, lambda1_fit, label="Fitted lambda")
        ax[0,1].scatter(Sz1, Y1, label="Observed intensity", color="red", alpha=0.5, marker="x")

        ax[1,1].plot(Sz1, x1_true, label="True intensity")
        ax[1,1].plot(Sz1, x1_fit, label="Fitted intensity")
        ax[1,1].fill_between(Sz1, x1_fit - 2 * np.sqrt(np.diag(Sigma1_fit)), x1_fit + 2 * np.sqrt(np.diag(Sigma1_fit)), alpha=0.2)
        ax[1,1].scatter(Sz1, np.log(Y1 +1), label="Log observarions", color="red", alpha=0.5, marker="x")
        ax[1,1].plot(Sz2, x2_pred, label="Predicted intensity")
        ax[1,1].fill_between(Sz2, x2_pred - 2 * np.sqrt(np.diag(Sigma2_pred)), x2_pred + 2 * np.sqrt(np.diag(Sigma2_pred)), alpha=0.2)
        ax[1,1].plot(Sz2, x2_true, label="True intensity")

        # Plot the third part of the diving AUV
        ax[0,2].plot(Szall, lambdaall_true, label="True intensity", c="green")
        ax[0,2].plot(Sz1, lambda1_fit, label="Fitted lambda", c="red")
        ax[0,2].plot(Szall, lambdaall_fit, c="red")
        ax[0,2].scatter(Sz1, Y1, label="Observed intensity", color="red", alpha=0.5, marker="x")
        ax[0,2].scatter(Sz2, Y2, color="red", alpha=0.5, marker="x")

        ax[1,2].plot(Sz1, x1_true, label="True intensity", c="green")
        ax[1,2].plot(Sz2, x2_true, c="green")
        ax[1,2].plot(Szall, xall_fit, c="red", label = "Fitted intensity")
        ax[1,2].plot(Szall, xall_prior, c="blue", label = "Prior intensity")
        ax[1,2].fill_between(Szall, xall_fit - 2 * np.sqrt(np.diag(Sigmaall_fit)), xall_fit + 2 * np.sqrt(np.diag(Sigmaall_fit)), alpha=0.2)


        
        for i in range(2):
            for j in range(3):
                ax[i,j].set_xlabel("Depth")
                ax[i,j].legend()

        plt.savefig("figures/tests/Model/diving_AUV.png")
        plt.close()

        # plotting only the first part of the diving AUV
        fig, ax = plt.subplots(1,2, figsize=(15,7))

        ax[0].plot(Sz1, lambda1_true, label="True intensity")
        ax[0].plot(Sz1, lambda1_pred, label="Predicted intensity")
        ax[0].scatter(Sz1, Y1, label="Observed intensity", color="red", alpha=0.5, marker="x")

        ax[1].plot(Sz1, x1_true, label="True intensity")
        ax[1].plot(Sz1, x1_pred, label="Predicted intensity")
        ax[1].plot(Sx1, x1_prior, label="Prior intensity")
        ax[1].fill_between(Sz1, x1_pred - 2 * np.sqrt(np.diag(Sigma1_pred)), x1_pred + 2 * np.sqrt(np.diag(Sigma1_pred)), alpha=0.2)
        ax[1].scatter(Sz1, np.log(Y1 +1), label="Log observarions", color="red", alpha=0.5, marker="x")

        plt.savefig("figures/tests/Model/diving_AUV_first_part.png")
        plt.close()


    def plot_down_sampling_values():
        pass


    def plot_adding_consecutive_paths(self, field, model, path_planner , n = 20):

        # Getting a start point
        sx = np.random.uniform(0, 2000)
        sy = np.random.uniform(0, 2000)
        sz = np.random.uniform(0, 50)
        s_current = np.array([sx, sy, sz])
        t_current = np.random.uniform(0, 24*3600)


        for i in range(n):
            # Get the path from the path planner
            path = path_planner.get_random_path(s_current, t_current, 100)
            S = path["S"]
            T = path["T"]
            Y = field.get_observation_Y(S, T)
            x_true = field.get_intensity_x(S, T)
            lambda_true = np.exp(x_true)

            # Add the values to the model
            t1 = time.time()
            model.add_new_values(S, T, Y)
            update_time = time.time() - t1

            s_current = S[-1]
            t_current = T[-1]

            n_points = len(T)
            total_points = len(model.data_dict["T"])
            total_time_of_operation = (model.data_dict["T"][-1] - model.data_dict["T"][0]) / 60


            print(f"i: {update_time:.2f} s adding {n_points} to the model, total points: {total_points}. Total time of operation: {total_time_of_operation:.2f} min")


        # Get the values from the model
        S = model.data_dict["S"]
        T = model.data_dict["T"]
        Y = model.data_dict["y"]
        x_fit = model.data_dict["x"]
        lambda_fit = np.exp(x_fit)  
        mu_prior = model.data_dict["mu"]  
        Sigma_fit = model.data_dict["Sigma"]

        # Get the true values
        x_true = field.get_intensity_x(S, T)
        lambda_true = np.exp(x_true)

        # Plot the path    
        fig, ax = plt.subplots(1, 2, figsize=(7,15))

        ax[0].plot(T, lambda_true, label="True intensity")
        ax[0].plot(T, lambda_fit, label="Fitted intensity")
        ax[0].scatter(T, Y, label="Observed intensity", color="red", alpha=0.5, marker="x")

        ax[1].plot(T, x_true, label="True intensity")
        ax[1].plot(T, x_fit, label="Fitted intensity")
        ax[1].plot(T, mu_prior, label="Prior intensity")
        ax[1].fill_between(T, x_fit - 2 * np.sqrt(np.diag(Sigma_fit)), x_fit + 2 * np.sqrt(np.diag(Sigma_fit)), alpha=0.2)
        ax[1].scatter(T, np.log(Y +1), label="Log observarions", color="red", alpha=0.5, marker="x")

        plt.savefig("figures/tests/Model/adding_consecutive_paths.png")
        plt.close()


    def plot_timing_data(self, model, name=""):
        timing_dict = model.timing_data

        for key in timing_dict.keys():

            timing_dat = timing_dict[key]
            x = np.linspace(0,1, len(timing_dat["time_list"]))
            plt.plot(x, timing_dat["time_list"], label = key)

        plt.legend()
        plt.title("time per call for function")
        plt.ylabel("Time seconds")

        plt.savefig(f"figures/tests/Model/timing_{name}.png")
        plt.close()


        for key in timing_dict.keys():
            
            timing_dat = timing_dict[key]
            x = np.linspace(0,1, len(timing_dat["time_list"]))
            y = np.cumsum(timing_dat["time_list"])
            plt.plot(x, y, label = key)

        plt.legend()
        plt.title("Cumulative time for function")
        plt.ylabel("Time seconds")
        plt.savefig(f"figures/tests/Model/cumulative_timing_{name}.png")
        plt.close()









    
    