
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import time
import itertools


class FieldPlotting:
    def __init__(self, field,
                 print_while_running=True):
        self.field = field
        self.print_while_running = print_while_running

    def plot(self, ax, **kwargs):
        raise NotImplementedError("This method must be implemented by a subclass")

    def plot_3d(self, ax, **kwargs):
        raise NotImplementedError("This method must be implemented by a subclass")
    
    @staticmethod
    def plot_zt_slize(field, sx, sy):
        S_lim = field.field["S_lim"]
        T_lim = field.field["T_lim"]
        Sz = np.linspace(S_lim[2][0],S_lim[2][1],100)
        T = np.linspace(T_lim[0],T_lim[1],100)
        Sz, T = np.meshgrid(Sz,T)
        Sz = Sz.flatten()
        T = T.flatten()
        S = np.array([np.repeat(sy,10000),np.repeat(sx,10000),Sz]).T
        x = field.interpolate_field(S,T)
        plt.scatter(Sz,T,c=x)
        plt.colorbar()
        plt.xlabel("z")
        plt.ylabel("t")
        plt.title(f"x = {sx}, y = {sy}")
        plt.savefig("figures/tests/Field/zt_slice.png")
        plt.close()

    

    def plot_6_slices(self, field):
        t1 = time.time()
        fig, ax = plt.subplots(2,3, figsize=(15,10))

        S_lim = field.field["S_lim"]
        T_lim = field.field["T_lim"]

        # Get the number of points in each direction 
        ny = field.field["n_s"][0]
        nx = field.field["n_s"][1]
        nz = field.field["n_s"][2]
        nt = field.field["n_t"]


        # Draw the constant values
        sy_const = np.random.uniform(S_lim[0][0],S_lim[0][1])
        sx_const = np.random.uniform(S_lim[1][0],S_lim[1][1])
        sz_const = np.random.uniform(S_lim[2][0],S_lim[2][1])
        t_const = np.random.uniform(T_lim[0],T_lim[1])

        # Get the sequences 
        n = 100
        Sy_grid = np.linspace(S_lim[0][0],S_lim[0][1],n)
        Sx_grid = np.linspace(S_lim[1][0],S_lim[1][1],n)
        Sz_grid = np.linspace(S_lim[2][0],S_lim[2][1],n)
        T_grid = np.linspace(T_lim[0],T_lim[1],n)

        # Get the repeated values
        Sy_rep = np.repeat(sy_const,n * n)
        Sx_rep = np.repeat(sx_const,n * n)
        Sz_rep = np.repeat(sz_const,n * n)
        T_rep = np.repeat(t_const,n * n)


        # plot the field for fixed values of y and  and random values of z and t
        Sz_mesh, T_mesh = np.meshgrid(Sz_grid,T_grid)
        Sz_mesh, T_mesh = Sz_mesh.flatten(), T_mesh.flatten()
        S = np.array([Sy_rep,Sx_rep,Sz_mesh]).T
        x = field.interpolate_field_CN(S,T_mesh)
        ax[0,0].scatter(Sz_mesh,T_mesh,c=x)
        ax[0,0].axvline(x=sz_const, color='r', linestyle='--')
        ax[0,0].axhline(y=t_const, color='r', linestyle='--')
        ax[0,0].set_xlabel("z")
        ax[0,0].set_ylabel("t")
        ax[0,0].set_title(f"x = {Sx_rep[0]:.1}, y = {Sy_rep[0]:.1}")
 

        # plot the field for fixed values of x and z and random values of y and t
        Sy_mesh, T_mesh = np.meshgrid(Sy_grid,T_grid)
        Sy_mesh, T_mesh = Sy_mesh.flatten(), T_mesh.flatten()
        S = np.array([Sy_mesh,Sx_rep,Sz_rep]).T
        x = field.interpolate_field_CN(S,T_mesh)
        ax[0,1].scatter(Sy_mesh,T_mesh,c=x)
        ax[0,1].axvline(x=sy_const, color='r', linestyle='--')
        ax[0,1].axhline(y=t_const, color='r', linestyle='--')
        ax[0,1].set_xlabel("y")
        ax[0,1].set_ylabel("t")
     
        # plot the field for fixed values of y and z and random values of x and t
        Sx_mesh, T_mesh = np.meshgrid(Sx_grid,T_grid)
        Sx_mesh, T_mesh = Sx_mesh.flatten(), T_mesh.flatten()
        S = np.array([Sy_rep,Sx_mesh,Sz_rep]).T
        x = field.interpolate_field_CN(S,T_mesh)
        ax[0,2].scatter(Sx_mesh,T_mesh,c=x)
        ax[0,2].axvline(x=sx_const, color='r', linestyle='--')
        ax[0,2].axhline(y=t_const, color='r', linestyle='--')
        ax[0,2].set_xlabel("x")
        ax[0,2].set_ylabel("t")
        ax[0,2].set_title(f"y = {Sy_rep[0]:.1}, z = {Sz_rep[0]:.1}")
     
        # plot the field for fixed values of y and t and random values of x and z
        Sx_mesh, Sz_mesh = np.meshgrid(Sx_grid,Sz_grid)
        Sx_mesh, Sz_mesh = Sx_mesh.flatten(), Sz_mesh.flatten()
        S = np.array([Sy_rep,Sx_mesh,Sz_mesh]).T
        x = field.interpolate_field_CN(S,T_rep)
        ax[1,0].scatter(Sx_mesh,Sz_mesh,c=x)
        ax[1,0].axvline(x=sx_const, color='r', linestyle='--')
        ax[1,0].axhline(y=sz_const, color='r', linestyle='--')
        ax[1,0].set_xlabel("x")
        ax[1,0].set_ylabel("z")
        ax[1,0].set_title(f"y = {Sy_rep[0]:.1}, t = {T_rep[0]:.1}")



        # plot the field for fixed values of x and t and random values of y and z
        Sy_mesh, Sz_mesh = np.meshgrid(Sy_grid,Sz_grid)
        Sy_mesh, Sz_mesh = Sy_mesh.flatten(), Sz_mesh.flatten()
        S = np.array([Sy_mesh,Sx_rep,Sz_mesh]).T
        x = field.interpolate_field_CN(S, T_rep)
        ax[1,1].scatter(Sy_mesh,Sz_mesh,c=x)
        ax[1,1].axvline(x=sy_const, color='r', linestyle='--')
        ax[1,1].axhline(y=sz_const, color='r', linestyle='--')
        ax[1,1].set_xlabel("y")
        ax[1,1].set_ylabel("z")
        ax[1,1].set_title(f"x = {Sx_rep[0]:.1}, t = {T_rep[0]:.1}")

        # plot the field for fixed values of z and t and random values of x and y
        Sy_mesh, Sx_mesh = np.meshgrid(Sy_grid,Sx_grid)
        Sy_mesh, Sx_mesh = Sy_mesh.flatten(), Sx_mesh.flatten()
        S = np.array([Sy_mesh,Sx_mesh,Sz_rep]).T
        x = field.interpolate_field_CN(S,T_rep)
        ax[1,2].scatter(Sy_mesh,Sx_mesh,c=x)
        ax[1,2].axvline(x=sy_const, color='r', linestyle='--')
        ax[1,2].axhline(y=sx_const, color='r', linestyle='--')
        ax[1,2].set_xlabel("y")
        ax[1,2].set_ylabel("x")
        ax[1,2].set_title(f"z = {Sz_rep[0]:.1}, t = {T_rep[0]:.1}")


        # Save and close
        fig.savefig("figures/tests/Field/6_slices_CN.png")
        plt.close()

        t2 = time.time()
        if self.print_while_running:
            print("Time to plot 6 CN slices:", t2 - t1)

    
    def plot_lambda_slices(self, field):
        t1 = time.time()
        fig, ax = plt.subplots(2,3, figsize=(15,10))

        S_lim = field.field["S_lim"]
        T_lim = field.field["T_lim"]


        n_s = field.field["n_s"]
        n_t = field.field["n_t"]
        grid_data = {"y": {"n": n_s[0], "lim": S_lim[0]},
                     "x": {"n": n_s[1], "lim": S_lim[1]},
                     "z": {"n": n_s[2], "lim": S_lim[2]},
                     "t": {"n": n_t, "lim": T_lim}}
        grid_keys = ["y","x","z","t"]
        
        n = 100
        for key in grid_data.keys():
            lim = grid_data[key]["lim"]
            grid_data[key]["const"] = np.random.uniform(lim[0],lim[1])
            grid_data[key]["grid"] = np.linspace(lim[0],lim[1],n)
            grid_data[key]["rep"] = np.repeat(grid_data[key]["const"],n * n)

        comb_2_list = list(itertools.combinations(grid_data.keys(),2))
        for i, comb in enumerate(comb_2_list):
            key1, key2 = comb # these are the ones that are varied
            key3, key4 = [key for key in grid_data.keys() if key not in comb] # these are kept constant
            all_keys = [key1,key2,key3,key4]
            key1_data = grid_data[key1]
            key2_data = grid_data[key2]
            key3_data = grid_data[key3]
            key4_data = grid_data[key4]
            key3_rep = key3_data["rep"]
            key4_rep = key4_data["rep"]
            key1_grid = key1_data["grid"]
            key2_grid = key2_data["grid"]
            key1_mesh, key2_mesh = np.meshgrid(key1_grid,key2_grid)
            key1_mesh, key2_mesh  = key1_mesh.flatten(), key2_mesh.flatten()
            S = np.zeros((len(key1_mesh),4))
            # Need to constuct the S matrix S = ("y","x","z","t")
            S[:,grid_keys.index(key1)] = key1_mesh
            S[:,grid_keys.index(key2)] = key2_mesh
            S[:,grid_keys.index(key3)] = key3_rep
            S[:,grid_keys.index(key4)] = key4_rep
            x = field.interpolate_field(S[:,:3],S[:,3])
            lam = field.get_mean_lambda_from_x(x)
            ax[i//3,i%3].scatter(key1_mesh,key2_mesh,c=lam)
            ax[i//3,i%3].axvline(x=key1_data["const"], color='r', linestyle='--')
            ax[i//3,i%3].axhline(y=key2_data["const"], color='r', linestyle='--')
            ax[i//3,i%3].set_xlabel(key1)
            ax[i//3,i%3].set_ylabel(key2)
            ax[i//3,i%3].set_title(f"{key1} = {key1_data['const']:.1f}, {key2} = {key2_data['const']:.1f}")


        plt.savefig("figures/tests/Field/lambda_slices.png")
        plt.close()

        t2 = time.time()
        if self.print_while_running:
            print("Time to plot 6 slices:", t2 - t1)


    def plot_marginal_slices(self, field):
        t1 = time.time()
        fig, ax = plt.subplots(2,3, figsize=(15,10))

        S_lim = field.field["S_lim"]
        T_lim = field.field["T_lim"]


        n_s = field.field["n_s"]
        n_t = field.field["n_t"]
        grid_data = {"y": {"n": n_s[0], "lim": S_lim[0]},
                     "x": {"n": n_s[1], "lim": S_lim[1]},
                     "z": {"n": n_s[2], "lim": S_lim[2]},
                     "t": {"n": n_t, "lim": T_lim}}
        grid_keys = ["y","x","z","t"]
        
        n = 100
        for key in grid_data.keys():
            lim = grid_data[key]["lim"]
            grid_data[key]["const"] = np.random.uniform(lim[0],lim[1])
            grid_data[key]["grid"] = np.linspace(lim[0],lim[1],n)
            grid_data[key]["rep"] = np.repeat(grid_data[key]["const"],n * n)

        comb_2_list = list(itertools.combinations(grid_data.keys(),2))
        for i, comb in enumerate(comb_2_list):
            key1, key2 = comb


        
        plt.savefig("figures/tests/Field/marginal_slices.png")
        plt.close()

        t2 = time.time()
        if self.print_while_running:
            print("Time to plot 6 slices:", t2 - t1)
        



    def plot_6_slices2(self, field):
        t1 = time.time()
        fig, ax = plt.subplots(2,3, figsize=(15,10))

        n_s = field.field["n_s"]
        n_t = field.field["n_t"]
        fig.suptitle(f"6 slices of the field \n n_y={n_s[0]}, n_x={n_s[1]}, n_z={n_s[2]}, n_t={n_t}")

        S_lim = field.field["S_lim"]
        T_lim = field.field["T_lim"]


        # Draw the constant values
        sy_const = np.random.uniform(S_lim[0][0],S_lim[0][1])
        sx_const = np.random.uniform(S_lim[1][0],S_lim[1][1])
        sz_const = np.random.uniform(S_lim[2][0],S_lim[2][1])
        t_const = np.random.uniform(T_lim[0],T_lim[1])

        # Get the sequences 
        n = 100
        Sy_grid = np.linspace(S_lim[0][0],S_lim[0][1],n)
        Sx_grid = np.linspace(S_lim[1][0],S_lim[1][1],n)
        Sz_grid = np.linspace(S_lim[2][0],S_lim[2][1],n)
        T_grid = np.linspace(T_lim[0],T_lim[1],n)

        # Get the repeated values
        Sy_rep = np.repeat(sy_const,n * n)
        Sx_rep = np.repeat(sx_const,n * n)
        Sz_rep = np.repeat(sz_const,n * n)
        T_rep = np.repeat(t_const,n * n)


        # plot the field for fixed values of y and  and random values of z and t
        Sz_mesh, T_mesh = np.meshgrid(Sz_grid,T_grid)
        Sz_mesh, T_mesh = Sz_mesh.flatten(), T_mesh.flatten()
        S = np.array([Sy_rep,Sx_rep,Sz_mesh]).T
        x = field.interpolate_field(S,T_mesh)
        ax[0,0].scatter(Sz_mesh,T_mesh,c=x)
        ax[0,0].axvline(x=sz_const, color='r', linestyle='--')
        ax[0,0].axhline(y=t_const, color='r', linestyle='--')
        ax[0,0].set_xlabel("z")
        ax[0,0].set_ylabel("t")
        ax[0,0].set_title(f"x = {Sx_rep[0]:.1}, y = {Sy_rep[0]:.1}")
 

        # plot the field for fixed values of x and z and random values of y and t
        Sy_mesh, T_mesh = np.meshgrid(Sy_grid,T_grid)
        Sy_mesh, T_mesh = Sy_mesh.flatten(), T_mesh.flatten()
        S = np.array([Sy_mesh,Sx_rep,Sz_rep]).T
        x = field.interpolate_field(S,T_mesh)
        ax[0,1].scatter(Sy_mesh,T_mesh,c=x)
        ax[0,1].axvline(x=sy_const, color='r', linestyle='--')
        ax[0,1].axhline(y=t_const, color='r', linestyle='--')
        ax[0,1].set_xlabel("y")
        ax[0,1].set_ylabel("t")
     
        # plot the field for fixed values of y and z and random values of x and t
        Sx_mesh, T_mesh = np.meshgrid(Sx_grid,T_grid)
        Sx_mesh, T_mesh = Sx_mesh.flatten(), T_mesh.flatten()
        S = np.array([Sy_rep,Sx_mesh,Sz_rep]).T
        x = field.interpolate_field(S,T_mesh)
        ax[0,2].scatter(Sx_mesh,T_mesh,c=x)
        ax[0,2].axvline(x=sx_const, color='r', linestyle='--')
        ax[0,2].axhline(y=t_const, color='r', linestyle='--')
        ax[0,2].set_xlabel("x")
        ax[0,2].set_ylabel("t")
        ax[0,2].set_title(f"y = {Sy_rep[0]:.1}, z = {Sz_rep[0]:.1}")
     
        # plot the field for fixed values of y and t and random values of x and z
        Sx_mesh, Sz_mesh = np.meshgrid(Sx_grid,Sz_grid)
        Sx_mesh, Sz_mesh = Sx_mesh.flatten(), Sz_mesh.flatten()
        S = np.array([Sy_rep,Sx_mesh,Sz_mesh]).T
        x = field.interpolate_field(S,T_rep)
        ax[1,0].scatter(Sx_mesh,Sz_mesh,c=x)
        ax[1,0].axvline(x=sx_const, color='r', linestyle='--')
        ax[1,0].axhline(y=sz_const, color='r', linestyle='--')
        ax[1,0].set_xlabel("x")
        ax[1,0].set_ylabel("z")
        ax[1,0].set_title(f"y = {Sy_rep[0]:.1}, t = {T_rep[0]:.1}")



        # plot the field for fixed values of x and t and random values of y and z
        Sy_mesh, Sz_mesh = np.meshgrid(Sy_grid,Sz_grid)
        Sy_mesh, Sz_mesh = Sy_mesh.flatten(), Sz_mesh.flatten()
        S = np.array([Sy_mesh,Sx_rep,Sz_mesh]).T
        x = field.interpolate_field(S, T_rep)
        ax[1,1].scatter(Sy_mesh,Sz_mesh,c=x)
        ax[1,1].axvline(x=sy_const, color='r', linestyle='--')
        ax[1,1].axhline(y=sz_const, color='r', linestyle='--')
        ax[1,1].set_xlabel("y")
        ax[1,1].set_ylabel("z")
        ax[1,1].set_title(f"x = {Sx_rep[0]:.1}, t = {T_rep[0]:.1}")

        # plot the field for fixed values of z and t and random values of x and y
        Sy_mesh, Sx_mesh = np.meshgrid(Sy_grid,Sx_grid)
        Sy_mesh, Sx_mesh = Sy_mesh.flatten(), Sx_mesh.flatten()
        S = np.array([Sy_mesh,Sx_mesh,Sz_rep]).T
        x = field.interpolate_field(S,T_rep)
        ax[1,2].scatter(Sy_mesh,Sx_mesh,c=x)
        ax[1,2].axvline(x=sy_const, color='r', linestyle='--')
        ax[1,2].axhline(y=sx_const, color='r', linestyle='--')
        ax[1,2].set_xlabel("y")
        ax[1,2].set_ylabel("x")
        ax[1,2].set_title(f"z = {Sz_rep[0]:.1}, t = {T_rep[0]:.1}")


        # Save and close
        fig.savefig("figures/tests/Field/6_slices2.png")
        plt.close()

        t2 = time.time()
        if self.print_while_running:
            print("Time to plot 6 slices:", round(t2 - t1,2), " s")

    @staticmethod
    def plot_covariance_function_illustration(field):
        # Testing how well the different covariance functions work

        S_lim = field.field["S_lim"]
        T_lim = field.field["T_lim"]

        Sy_const = np.random.uniform(S_lim[0][0],S_lim[0][1])
        Sx_const = np.random.uniform(S_lim[1][0],S_lim[1][1])
        Sz_const = np.random.uniform(S_lim[2][0],S_lim[2][1])
        St_const = np.random.uniform(T_lim[0],T_lim[1])
        n = 1000
        Sy_random = np.random.uniform(S_lim[0][0],S_lim[0][1],n)
        Sx_random = np.random.uniform(S_lim[1][0],S_lim[1][1],n)
        Sz_random = np.random.uniform(S_lim[2][0],S_lim[2][1],n)
        St_random = np.random.uniform(T_lim[0],T_lim[1],n)
        Sy_rep = np.repeat(Sy_const,n)
        Sx_rep = np.repeat(Sx_const,n)
        Sz_rep = np.repeat(Sz_const,n)
        St_rep = np.repeat(St_const,n)

        fig, ax = plt.subplots(1,3,figsize=(15,5))
        # First we check the spatial correlation
        # This is done by varying the z value and keeping the other values constant
        S = np.array([Sy_rep,Sx_rep,Sz_random]).T
        T = St_rep
        cov_matrix = field.get_covariance_matrix(S,T)
        dist = field.distance_matrix_one_dimension(Sz_random,Sz_random)
        dist1 = dist[0,:]
        ax[0].scatter(dist1,cov_matrix[0,:], s=1)
        ax[0].set_xlabel(r"h_z")
        phi_spatial_z = field.covariance_parameters["phi_spatial_z"]
        ax[0].set_title(f"z correlation phi_z = {phi_spatial_z:.2f}")

        # Next we check the yx-plane correlation
        # This is done by varying the y and x value and keeping the other values constant
        S = np.array([Sy_random,Sx_random,Sz_rep]).T
        T = St_rep
        cov_matrix = field.get_covariance_matrix(S,T)
        dist = distance_matrix(S[:,:2],S[:,:2])
        dist1 = dist[0,:]
        ax[1].scatter(dist1,cov_matrix[0,:], s=1)
        ax[1].set_xlabel(r"h_yx")
        phi_spatial_yx = field.covariance_parameters["phi_spatial_yx"]
        ax[1].set_title(f"yx correlation phi_yx = {phi_spatial_yx:.2f}")

        # Finally we check the temporal correlation
        # This is done by varying the t value and keeping the other values constant
        S = np.array([Sy_rep,Sx_rep,Sz_rep]).T
        T = St_random
        cov_matrix = field.get_covariance_matrix(S,T)
        dist = field.distance_matrix_one_dimension(St_random,St_random)
        dist1 = dist[0,:]
        ax[2].scatter(dist1,cov_matrix[0,:], s=1)
        ax[2].set_xlabel(r"h_t")
        phi_temporal = field.covariance_parameters["phi_temporal"]
        ax[2].set_title(f"temporal correlation phi_t = {phi_temporal:.2f}")

        plt.savefig("figures/tests/Field/covariance_matrix.png")
        plt.close()

    @staticmethod
    def plot_different_corr_parameters(field, phi_yx_range, phi_z_range, phi_t_range):
        fig, ax = plt.subplots(1,3,figsize=(15,5))

        S_lim = field.field["S_lim"]
        T_lim = field.field["T_lim"]

        for phi_spatial_z in phi_z_range:
            field.field["phi_spatial_z"] = phi_spatial_z
            h_z = np.linspace(S_lim[2][0],S_lim[2][1],100)
            ax[0].plot(h_z,field.get_z_correlation(h_z),label=f"phi_z = {phi_spatial_z:.2f}")
        ax[0].set_title("z correlation")
        ax[0].set_xlabel("h_z")
        ax[0].set_ylabel("Correlation")
        ax[0].legend()

        for phi_spatial_yx in phi_yx_range:
            field.field["phi_spatial_yx"] = phi_spatial_yx
            h_yx = np.linspace(0,2000,100)
            ax[1].plot(h_yx,field.get_yx_correlation(h_yx),label=f"phi_yx = {phi_spatial_yx:.2f}")
        ax[1].set_title("yx correlation")
        ax[1].set_xlabel("h_yx")
        ax[1].set_ylabel("Correlation")
        ax[1].legend()

        for phi_temporal in phi_t_range:
            field.field["phi_temporal"] = phi_temporal
            h_t = np.linspace(0,24*3600,100)
            ax[2].plot(h_t,field.get_temporal_correlation(h_t),label=f"phi_t = {phi_temporal:.2f}")
        ax[2].set_title("temporal correlation")
        ax[2].set_xlabel("h_t")
        ax[2].set_ylabel("Correlation")
        ax[2].legend()

        plt.savefig("figures/tests/Field/correlation_parameters.png")
        plt.close()
    
    def plot_prior_intensity(self, field ,n = 10, m= 100):
        S_lim = field.field["S_lim"]
        T_lim = field.field["T_lim"]

        Sy = np.repeat(np.random.uniform(S_lim[0][0],S_lim[0][1]),m)
        Sx = np.repeat(np.random.uniform(S_lim[1][0],S_lim[1][1]),m)
        Sz = np.linspace(S_lim[2][0],S_lim[2][1],m)
        St = np.linspace(T_lim[0],T_lim[1],n)
        for t in St:
            intensity = field.get_intensity_mu_x(np.array([Sy,Sx,Sz]).T,np.repeat(t,m))
            plt.plot(Sz,intensity, label=f"t = {t:.0f}")
        plt.xlabel("Depth [m]")
        plt.ylabel("Intensity")
        plt.title("Prior intensity function")
        plt.legend()
        plt.savefig("figures/tests/Field/prior_intensity.png")
        plt.close()


        
    