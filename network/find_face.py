from __future__ import absolute_import, division, print_function
import tensorflow as tf
import model
import utils.image as img
import pandas as pd


tf.enable_eager_execution()
model.initialize_gpu()

MODEL_NAME = './model/location/07_22_1.h5'


def prepare_image(image_name):
    raw_image = open(image_name, 'rb').read()
    return tf.io.decode_image(raw_image)


def find_face(image_name):
    image = prepare_image(image_name)
    image_reshaped = tf.expand_dims(image, 0)
    location_model = tf.keras.models.load_model(MODEL_NAME)
    return location_model.predict(image_reshaped)


result = find_face("../data/Selfie-dataset/balint_crop.jpg")[0]
print(result*306)
img.draw_rect_on_image("../balint_crop.jpg", result[0]*306, result[1]*306, result[2]*306, result[3]*306)
# df = pd.read_csv('../data/result.csv')
# df = df.loc[df['image_name'] == '02ee8060b02f11e38fe012b1c8928cc9_6']
# print(df)
