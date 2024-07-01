
import os
import numpy as np
import matplotlib.pyplot as plt


class ModelLogNormalPlotting:
    def __init__(self, model):
        self.model = model

        self.figures_path = "figures/tests/ModelLogNormal/"

        # Create the folder
        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path)


    def plot_diving_AUV(self, field):

        "Plotting a diving AUV in the field"
        pass



    