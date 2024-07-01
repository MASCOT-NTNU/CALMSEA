import numpy as np


# file path
file_path = 'src/echosounder_files/prior_latest.npz'

loaded_data = {}
with np.load(file_path) as data:

    files = data.files

    for f in files:
        loaded_data[f] = data[f]


for key in loaded_data.keys():
    print(key)
    print(loaded_data[key].shape)


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Plot the surface

meshgrid = np.array(loaded_data['meshgrid'])
upper_s_mean = loaded_data['upper_s_mean'].flatten()
lower_s_mean = loaded_data['lower_s_mean'].flatten()


def get_mean(data):
    upper_s_mean = data['upper_s_mean']
    lower_s_mean = data['lower_s_mean']
    mean_upper = np.mean(upper_s_mean, axis=0)
    mean_lower = np.mean(lower_s_mean, axis=0)
    mean = np.mean(np.array([mean_upper, mean_lower]), axis=0)
    return mean

def get_patch_width(data):
    upper_s_mean = data['upper_s_mean']
    lower_s_mean = data['lower_s_mean']
    mean = get_mean(data)
    
    diff = []
    for i in range(upper_s_mean.shape[0]):
        if np.abs(upper_s_mean[i] - mean) > 0.1 and np.abs(lower_s_mean[i] - mean) > 0.1:
            diff.append(np.abs(upper_s_mean[i] - lower_s_mean[i]))
    patch_width = np.mean(diff)
    return patch_width



def get_value_S(data, S):
    # This function returns 1 if s is inside the patch and 0 otherwise
    # S is a list of points in (x,y,z) coordinates
    upper_s_mean = data['upper_s_mean']
    lower_s_mean = data['lower_s_mean']
    mean = get_mean(data)

    meshgrid = np.array(data['meshgrid'])
    x = meshgrid[:,0]
    y = meshgrid[:,1]

    # Get limits from the data
    x_lim = [np.min(x), np.max(x)]
    y_lim = [np.min(y), np.max(y)]

    valures = np.zeros(len(S))
    for i, s in enumerate(S):
        x_s = s[0]
        y_s = s[1]
        z_s = s[2]
        # Get closest 
        if z_s > mean:
            z = upper_s_mean
        else:
            z = lower_s_mean
        valures[i] = 1 if z_s < z[x == x_s and y == y_s] else 0


    return S

print("heatmap", loaded_data["heat_map"])

meshgrid = loaded_data['meshgrid']
print("meshgrid", meshgrid)
print("meshgrid", meshgrid.shape)
x = np.array(meshgrid[:,0])
y = np.array(meshgrid[:,1])
print("x", x.shape)
print("y", y.shape)

patch_width = get_patch_width(loaded_data)
print("patch_width", patch_width)

mean = get_mean(loaded_data)
print("mean depth", mean)



x_coord = loaded_data['x_coords'][0]
y_coord = loaded_data['y_coords']
meshgrid = np.meshgrid(x_coord, y_coord)

print("meshgrid[0].shape", meshgrid[0].shape)  
print("meshgrid[1].shape", meshgrid[1].shape)
print("upper_s_mean.shape", upper_s_mean.shape)
print("lower_s_mean.shape", lower_s_mean.shape)



plt.plot(upper_s_mean)
plt.plot(lower_s_mean)
plt.show()



plt.figure()
plt.pcolormesh(meshgrid[0], meshgrid[1], upper_s_mean.reshape(meshgrid[0].shape))
plt.colorbar()
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf_high = ax.plot_surface(meshgrid[0], meshgrid[1], upper_s_mean.reshape(meshgrid[0].shape), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf_low = ax.plot_surface(meshgrid[0], meshgrid[1], lower_s_mean.reshape(meshgrid[0].shape), cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf_low, shrink=0.5, aspect=5)

plt.show()


