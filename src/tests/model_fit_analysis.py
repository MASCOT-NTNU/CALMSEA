import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns


# load the data
df = pd.read_csv("data/fitting_model_test/fitting_model.csv")


print(df.head(20))
print(df["method"].unique())

# plot the results


levels, categories = pd.factorize(df['method'])
colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

plt.scatter(df["n_old"] + df["n_new"], df["error"],  c=colors)
plt.xlabel("# points in model + # points added")
plt.ylabel("absolute error fitted model")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/error_absolute.png")
plt.close()

plt.scatter(df["n_old"] + df["n_new"], df["error_sd"], c=colors)
plt.xlabel("# points in model + # points added")
plt.ylabel("Sd")
plt.title("Error between the fitted and true x")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/error_sd.png")
plt.close()

plt.scatter(df["n_old"] + df["n_new"], df["len_norm"], c=colors)
plt.xlabel("# points in model + # points added")
plt.ylabel("Iterations to fit model")
plt.title("Iterations to fit the model vs total points")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/itter_fit_model_opn.png")
plt.close()

plt.scatter(df["n_new"], df["len_norm"], c=colors)
plt.xlabel("points added")
plt.ylabel("Iterations to fit model")
plt.title("Iterations to fit the model vs new points")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/itter_fit_model_n.png")
plt.close()

plt.scatter(df["n_old"], df["len_norm"],  c=colors)
plt.xlabel("# points in model")
plt.ylabel("Iterations to fit model")
plt.title("Iterations to fit the model vs old points")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/itter_fit_model_o.png")
plt.close()

time_pr_iteration = df["time"] / df["len_norm"]

plt.scatter(df["n_old"] + df["n_new"], time_pr_iteration,  c=colors)
plt.ylabel("Time pr iteration (s)")
plt.xlabel("# points in model + # points added")
plt.title("Time pr iteration")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/time_pr_iteration.png")
plt.close()


# time to fit the model
plt.scatter(df["n_old"] + df["n_new"], df["time"],  c=colors)
plt.ylabel("Time (s)")
plt.xlabel("# points in model + # points added")
plt.title("Time to fit the model")
plt.legend(handles=handles, title="Initial guess method")
plt.savefig("figures/fitting_model/time_fit_model.png")
plt.close()


# Plot multiple boxplots on one graph
fig, ax = plt.subplots(2,3, figsize=(15,15))

df["time_pr_iteration"] = df["time"] / df["len_norm"]


sns.boxenplot(x="method", y="error", data=df, ax=ax[0,0])
sns.boxenplot(x="method", y="error_sd", data=df, ax=ax[0,1])
sns.boxenplot(x="method", y="len_norm", data=df, ax=ax[0,2])
sns.boxenplot(x="method", y="time", data=df, ax=ax[1,0])
sns.boxenplot(x="method", y="time_pr_iteration", data=df, ax=ax[1,1])
sns.boxenplot(x="method", y="error_new", data=df, ax=ax[1,2])


plt.savefig("figures/fitting_model/boxplot.png")
plt.close()