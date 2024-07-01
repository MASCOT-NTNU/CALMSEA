
import pandas as pd
import numpy as np

from plot_silc_img import plot_h5_image


# file paths 
raw_stats_path = 'src/silc_data/'
raw_stats_names = ['RAW-STATS-testdive-3.csv',
                   'RAW-STATS_mausund_run1.csv',
                   'RAW-STATS_mausund_run2.csv']
silcam_images_path = '/Users/ajolaise/Documents/silcam_images/'
store_image_path = "silcam_images/large_major_axis/"
percentage = 0.01


for file in raw_stats_names:
    print(file)
    df = pd.read_csv(raw_stats_path + file)

    # Get the 1 % of the images with the largest major axis
    df = df.sort_values(by="major_axis_length", ascending=False)
    n_images = int(len(df)*percentage)
    good_images = df.head(n_images)


    for row in good_images.iterrows():
        image_name = row[1]["export name"]
        if image_name != "not_exported":
            print(image_name)
            img_id, parickle_id = image_name.split("-")
            print(img_id, parickle_id)
            filename = silcam_images_path + img_id + ".h5"

            major_axis = row[1]["major_axis_length"]
            extra_text = f"ma={major_axis:.3f}"

            try:
                plot_h5_image(store_image_path, filename, parickle_id, extra_text=extra_text)
            except:
                print("could not find image")