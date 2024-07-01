import numpy as np
import h5py
from PIL import Image


def plot_h5_image(store_image_path, image_file, particle_id, extra_text=""):

    with h5py.File(image_file, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 


         
        # get the object type for a_group_key: usually group or dataset 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[particle_id])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[particle_id])
        # preferred methods to get dataset values:
        ds_obj = f[particle_id]      # returns as a h5py dataset object
        ds_arr = f[particle_id][()]  # returns as a numpy array

        data = np.array(data)

        img  = Image.fromarray(data)
        # Saving the image
        name = image_file.split("/")[-1]
        if extra_text == "":
            img.save(f"{store_image_path}{name}_{particle_id}.png")
        else:
            img.save(f"{store_image_path}{name}_{particle_id}_{extra_text}.png")