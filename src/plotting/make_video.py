import imageio.v2 as imageio
import cv2

iter_i = 100

# Making a video
fileList = []
for i in range(iter_i-1):
    fileList.append("figures/simulation/depth_profile_" + str(i+1) + ".png",)


# Create a video from these images using cv2
img_array = []
for filename in fileList:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('figures/simulation/depth_profile_video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
out.release()
cv2.destroyAllWindows()
#writer = cv2.VideoWriter('myvideo.mp4',cv2.VideoWriter_fourcc(*'XVID'),fps,(width,height))
#writer.release()

"""
# Make a 1 minute video
choose_fps = round(iter_i / 60)
writer = imageio.get_writer("figures/simulation/depth_profile_video" + '.mp4')

for im in fileList:
    writer.append_data(imageio.imread(im, format="mp4"))
writer.close()
"""