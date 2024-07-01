from scipy import interpolate

# Create a 1D interpolation function
f = interpolate.interp1d([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])


# print methods for interpolate.RegularGridInterpolator(method)
