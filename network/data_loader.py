import tensorflow as tf
import numpy as np
import math
from model import model

BATCH_SIZE = 16
EPOCH_NUM = 10
TEST_SIZE = 500
SHUFFLE_BUFFER = 500
TOTAL_NUM = 5110
steps_per_epoch = int(math.floor((TOTAL_NUM - TEST_SIZE) / BATCH_SIZE))
data_file_name = 'labelled_selfies.tfrecord'

# Create a dictionary describing the features.
image_feature_description = {
    'raw_image': tf.FixedLenFeature([], tf.string),
    'image_name': tf.FixedLenFeature([], tf.string),
    'face_width': tf.FixedLenFeature([], tf.float32),
    'face_height': tf.FixedLenFeature([], tf.float32),

    'anger': tf.FixedLenFeature([], tf.int64),
    'contempt': tf.FixedLenFeature([], tf.int64),
    'disgust': tf.FixedLenFeature([], tf.int64),
    'fear': tf.FixedLenFeature([], tf.int64),
    'happiness': tf.FixedLenFeature([], tf.int64),
    'neutral': tf.FixedLenFeature([], tf.int64),
    'sadness': tf.FixedLenFeature([], tf.int64),
    'surprise': tf.FixedLenFeature([], tf.int64),
    'is_tongue_out': tf.FixedLenFeature([], tf.int64),
    'is_duck_face': tf.FixedLenFeature([], tf.int64),

    'face_cent_x': tf.FixedLenFeature([], tf.float32),
    'face_cent_y': tf.FixedLenFeature([], tf.float32),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    sample = tf.parse_single_example(example_proto, image_feature_description)
    sample['raw_image'] = tf.image.decode_image(sample['raw_image'])

    # Encoding of the output layer of the face detection phase (all input images are 306*306)
    sample['face_position_encoding'] = tf.convert_to_tensor(
                                        [sample['face_cent_x']/306, sample['face_cent_y']/306,
                                         sample['face_width']/306, sample['face_height']/306], np.float32)
    return sample


def _parse_into_input_tuple(example_proto):
    x_val = example_proto['raw_image']
    x_val.set_shape([306, 306, 3])
    y_val = example_proto['face_position_encoding']
    y_val.set_shape([4])
    return x_val, y_val


# Loads the dataset with the defined flags and returns an iterator
def load_data_set():
    dataset = tf.data.TFRecordDataset(data_file_name)
    dataset = dataset.map(_parse_image_function)  # Parse the record into tensors.
    dataset = dataset.map(_parse_into_input_tuple)
    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    test_dataset = dataset.take(TEST_SIZE)
    test_dataset = test_dataset.batch(1)
    train_dataset = dataset.skip(TEST_SIZE)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(BATCH_SIZE)
    return train_dataset, test_dataset


# Counting the no. of elements
def count_elements_in_dataset(iterator):
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        c = 0
        while True:
            try:
                sess.run(iterator.get_next())
                c += BATCH_SIZE
                if c % 100 == 0:
                    print("{} and counting...".format(c))
            except tf.errors.OutOfRangeError:
                print(c)
                break


# Display first element
def get_first_element_in_dataset(iterator):
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        data = sess.run(iterator.get_next())
        return data


(train, test) = load_data_set()
# print(get_first_element_in_dataset(train.make_initializable_iterator())[1].shape)
# model.summary()

model.fit(train, epochs=EPOCH_NUM, steps_per_epoch=steps_per_epoch)
score = model.evaluate(test, steps=1)
print("\nTest evaluation:")
print(score)
