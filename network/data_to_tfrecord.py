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

with tf.python_io.TFRecordWriter('labelled_selfies.tfrecord') as writer:
    # Filling up each list
    for index, row in df.iterrows():

        # Looking up JPG
        rel_file_name = "../data/Selfie-dataset/images/{}.jpg".format(row['image_name'])
        raw_image = open(rel_file_name, 'rb').read()

        # One-hot coding attributes
        attributes = row['anger':'is_duck_face']
        max_attr = max(attributes)

        # Creating TFRecord
        feature_dict = {
            'raw_image': _bytes_feature(raw_image),
            'image_name': _bytes_feature(row['image_name'].encode('utf-8')),
            'face_width': _float_feature(row['face_width']),
            'face_height': _float_feature(row['face_height']),

            'anger': _int64_feature(row['anger'] == max_attr),
            'contempt': _int64_feature(row['contempt'] == max_attr),
            'disgust': _int64_feature(row['disgust'] == max_attr),
            'fear': _int64_feature(row['fear'] == max_attr),
            'happiness': _int64_feature(row['happiness'] == max_attr),
            'neutral': _int64_feature(row['neutral'] == max_attr),
            'sadness': _int64_feature(row['sadness'] == max_attr),
            'surprise': _int64_feature(row['surprise'] == max_attr),
            'is_tongue_out': _int64_feature(row['is_tongue_out'] == max_attr),
            'is_duck_face': _int64_feature(row['is_duck_face'] == max_attr),

            'face_cent_x': _float_feature(row['face_cent_x']),
            'face_cent_y': _float_feature(row['face_cent_y']),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        # Write to disk
        writer.write(example.SerializeToString())

