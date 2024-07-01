import h5py
import numpy as np
from PIL import Image


file_path = 'silcam_images/'
name = "D20240605T104042.271393"
filename = file_path + name + ".h5"


with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    all_keys = list(f.keys())

    
    for img_id in all_keys:

        # get the object type for a_group_key: usually group or dataset 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[img_id])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[img_id])
        # preferred methods to get dataset values:
        ds_obj = f[img_id]      # returns as a h5py dataset object
        ds_arr = f[img_id][()]  # returns as a numpy array

        data = np.array(data)
        print("data", data.shape)

        img  = Image.fromarray(data)
        # Saving the image
        img.save(f"silcam_images/png_img/{name}_{img_id}.png")



print("data", data)
print(ds_obj)
print(ds_arr)

print(data[0].shape)

data = np.array(data)

print(data[:,:,2])
print(data.shape)

import matplotlib.pyplot as plt
import seaborn as sns

plt.imshow(data[:,:,0])
plt.show()




img  = Image.fromarray(data)
# Saving the image
img.save("silcam_images/png_img/Image_from_array.png")