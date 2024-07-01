import numpy as np
import pandas as pd
import h5py
from PIL import Image


# file paths 
raw_stats_path = 'src/silc_data/'
raw_stats_names = ['RAW-STATS-testdive-3.csv',
                   'RAW-STATS_mausund_run1.csv',
                   'RAW-STATS_mausund_run2.csv',
                   "RAW-STATS-cla_0606.csv",
                    "RAW-STATS-cla_0607.csv"]
silcam_images_path = '/Users/ajolaise/Documents/silcam_images/export/'
store_image_path = "silcam_images/best_copepod_images/"
threshold = 0.8

def plot_image(image_file, particle_id, extra_text=""):

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



for file in raw_stats_names:
    print(file)
    df = pd.read_csv(raw_stats_path + file)

    # get the images with higher than 0.97 probability of being a copepod
    good_images = df[df["probability_copepod"] > threshold]


    for row in good_images.iterrows():
        image_name = row[1]["export name"]
        probability = row[1]["probability_copepod"]
        img_id, parickle_id = image_name.split("-")
        print(img_id, parickle_id)
        filename = silcam_images_path + img_id + ".h5"

        prob = str(round(probability,4))  
        extra_text = f"p={prob}"

        try:
            plot_image(filename, parickle_id, extra_text=extra_text)
        except:
            pass
            print("could not find image")

    """
    for image_name in good_images["export name"]:
        img_id, parickle_id = image_name.split("-")
        print(img_id, parickle_id)
        filename = silcam_images_path + img_id + ".h5"

        try:
            plot_image(filename, parickle_id)
        except:
            print("could not find image")

    """
