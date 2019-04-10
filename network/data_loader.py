import tensorflow as tf

BATCH_SIZE = 10
REPEAT_INFINITELY = False
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
    return sample


# Loads the dataset with the defined flags and returns an iterator
def load_data_set():
    dataset = tf.data.TFRecordDataset(data_file_name)
    dataset = dataset.map(_parse_image_function)  # Parse the record into tensors.
    if REPEAT_INFINITELY:
        dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.make_initializable_iterator()


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
                    print("{} an counting...".format(c))
            except tf.errors.OutOfRangeError:
                print(c)
                break
