

import pandas as pd
import numpy as np
import os

from plot_silc_img import plot_h5_image

# file paths 
raw_stats_path = 'src/silc_data/'
raw_stats_names = ['RAW-STATS-testdive-3.csv',
                   'RAW-STATS_mausund_run1.csv',
                   'RAW-STATS_mausund_run2.csv',
                   "RAW-STATS-cla_0606.csv",
                    "RAW-STATS-cla_0607.csv"]
silcam_images_path = '/Users/ajolaise/Documents/silcam_images/export/'
store_image_path = "silcam_images/particel_examples/"


df = pd.read_csv(raw_stats_path + raw_stats_names[0])
print(df.columns)

particle_infos = [
    {"particle_type": "copepod", "prob_str": 'probability_copepod'},
    {"particle_type": "oil", "prob_str": "probability_oil"},
    {"particle_type": "bubble", "prob_str": "probability_bubble"},
    {"particle_type": "feacal_pellets", "prob_str": 'probability_faecal_pellets'},
    {"particle_type": "diatom_chain", "prob_str": 'probability_diatom_chain'},
    {"particle_type": "oily_gas", "prob_str": 'probability_oily_gas'},
]



threshold = 0.95


for file in raw_stats_names:
    print(file)
    df = pd.read_csv(raw_stats_path + file)

    for particle_info in particle_infos:

        # Get the 1 % of the images with the largest major axis
        good_images = df[df[particle_info["prob_str"]] > threshold]
        

        print(particle_info)
        print(len(good_images))

        # Shuffle the images
        good_images = good_images.sample(frac=1)
        stored = 0
        for row in good_images.iterrows():
            image_name = row[1]["export name"]
            if image_name != "not_exported":
                img_id, parickle_id = image_name.split("-")
                #print(img_id, parickle_id)
                filename = silcam_images_path + img_id + ".h5"

                probability = row[1][particle_info["prob_str"]]
                extra_text = f"p={probability:.3f}"

                store_path = store_image_path + particle_info["particle_type"] + "/"

                # Create the directory if it does not exist
                if not os.path.exists(store_path):
                    os.makedirs(store_path)


                try:
                    plot_h5_image(store_path,
                                filename,
                                parickle_id,
                                extra_text=extra_text)
                    stored += 1
                except:
                    #print("could not find image")
                    pass
        print(f"Stored {stored} images")