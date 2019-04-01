from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pandas as pd

tf.enable_eager_execution()

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


df = pd.read_csv('../data/result.csv')
print(df)

image_tensor_list = []
image_name_list = []
face_width_list = []
face_height_list = []
anger_list = []
contempt_list = []
disgust_list = []
fear_list = []
happiness_list = []
neutral_list = []
sadness_list = []
surprise_list = []
is_tongue_out_list = []
is_duck_face_list = []
face_cent_x_list = []
face_cent_y_list = []

# Filling up each list
for index, row in df.iterrows():

    rel_file_name = "../data/Selfie-dataset/images/{}.jpg".format(row['image_name'])

    # Looking up and parsing jpg
    img_raw = tf.io.read_file(rel_file_name)
    result = tf.image.decode_jpeg(img_raw, channels=3)
    image_tensor_list.append(tf.serialize_tensor(result))

    # Adding other features
    image_name_list.append(row['image_name'].encode('utf-8'))   # String -> bytes
    face_width_list.append(row['face_width'])
    face_height_list.append(row['face_height'])
    face_cent_x_list.append(row['face_cent_x'])
    face_cent_y_list.append(row['face_cent_y'])

    attributes = row['anger':'is_duck_face']
    max_attr = max(attributes)

    anger_list.append(row['anger'] == max_attr)
    contempt_list.append(row['contempt'] == max_attr)
    disgust_list.append(row['disgust'] == max_attr)
    fear_list.append(row['fear'] == max_attr)
    happiness_list.append(row['happiness'] == max_attr)
    neutral_list.append(row['neutral'] == max_attr)
    sadness_list.append(row['sadness'] == max_attr)
    surprise_list.append(row['surprise'] == max_attr)
    is_tongue_out_list.append(row['is_tongue_out'] == max_attr)
    is_duck_face_list.append(row['is_duck_face'] == max_attr)

# Creating TFRecord
feature_dict = {
    'image_tensor': _bytes_feature(image_tensor_list),
    'image_name': _bytes_feature(image_name_list),
    'face_width': _float_feature(face_width_list),
    'face_height': _float_feature(face_height_list),
    'anger': _int64_feature(anger_list),
    'contempt': _int64_feature(contempt_list),
    'disgust': _int64_feature(disgust_list),
    'fear': _int64_feature(fear_list),
    'happiness': _int64_feature(happiness_list),
    'neutral': _int64_feature(neutral_list),
    'sadness': _int64_feature(sadness_list),
    'surprise': _int64_feature(surprise_list),
    'is_tongue_out': _int64_feature(is_tongue_out_list),
    'is_duck_face': _int64_feature(is_duck_face_list),
    'face_cent_x': _float_feature(face_cent_x_list),
    'face_cent_y': _float_feature(face_cent_y_list),
}

example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

# Write to disk
with tf.python_io.TFRecordWriter('labelled_selfies.tfrecord') as writer:
    writer.write(example.SerializeToString())

