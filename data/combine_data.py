import pandas as pd

file_list = [
    "selected_0_999.csv",
    "selected_1000_1999.csv",
    "selected_2000_2999.csv",
    "selected_3000_3999.csv",
    "selected_4000_4999.csv",
    "selected_5000_5439.csv",
    "selected_5440_5999.csv",
    "selected_6000_6499.csv",
]

frames = [pd.read_csv(file) for file in file_list]
result = pd.concat(frames)

# Changing the coordinate system for faces
# Instead of top-left corner use centre
result = result.assign(face_cent_x=lambda x: x['face_left'] + x['face_width'] / 2.0)
result = result.assign(face_cent_y=lambda x: x['face_top'] + x['face_height'] / 2.0)

result = result.drop(columns=['face_top', 'face_left'])

result.to_csv("result.csv", index=None, header=True)
