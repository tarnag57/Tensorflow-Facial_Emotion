import tensorflow as tf


def initialize_gpu():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)


def get_location_model():
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Conv2D(
            input_shape=[306, 306, 3],
            filters=6,
            kernel_size=(5, 5),
            padding='same',
            activation=tf.nn.relu
        ),
        layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            padding='same',
            activation=tf.nn.relu
        ),
        layers.MaxPool2D(
            pool_size=(2, 2),
            strides=2
        ),
        layers.Conv2D(
            filters=12,
            kernel_size=(5, 5),
            padding='same',
            activation=tf.nn.relu
        ),
        layers.MaxPool2D(
            pool_size=(2, 2),
            strides=2
        ),
        layers.Conv2D(
            filters=12,
            kernel_size=(5, 5),
            padding='same',
            activation=tf.nn.relu
        ),
        layers.MaxPool2D(
            pool_size=(2, 2),
            strides=2
        ),
        layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            padding='same',
            activation=tf.nn.relu
        ),
        layers.MaxPool2D(
            pool_size=(2, 2),
            strides=2
        ),
        layers.Flatten(),
        layers.Dense(
            units=256,
            activation=tf.nn.relu
        ),
        layers.Dropout(
            rate=0.05
        ),
        layers.Dense(
            units=4,
            activation=tf.nn.sigmoid
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9, lr=0.002),
        loss=tf.keras.losses.mean_squared_error,
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    return model
