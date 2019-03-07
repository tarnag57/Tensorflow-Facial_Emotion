import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image
import numpy as np


root = "../data/Selfie-dataset/images/"

def draw_rect_on_image(file_name, centre_x, centre_y, width, height):
    im = np.array(Image.open(root + file_name), dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    rect = patches.Rectangle((centre_x - width / 2.0, centre_y - height / 2.0),
                             width, height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


# For testing
df = pd.read_csv("../data/result.csv")
row = df.iloc[2]
img_name = row["image_name"] + ".jpg"
draw_rect_on_image(img_name, row["face_cent_x"], row["face_cent_y"],
                   row["face_width"], row["face_height"])
