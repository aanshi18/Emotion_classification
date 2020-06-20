from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


directory = ['D:\Aanshi\AANSHI2020\clique\data\Training','D:\Aanshi\AANSHI2020\clique\data\img_augmented\Training'];

image_size = (64, 64)
batch_size = 32
patience = 50

for dir in directory:
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dir,
        labels="inferred",
        label_mode="categorical",
        class_names=["emotion0", "emotion1", "emotion2", "emotion3", "emotion4", "emotion5", "emotion6"],
        image_size=image_size,
        batch_size=batch_size,
        seed=123,
        validation_split=0.2,
        subset="training",

    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dir,
        labels="inferred",
        label_mode="categorical",
        class_names=["emotion0", "emotion1", "emotion2", "emotion3", "emotion4", "emotion5", "emotion6"],
        image_size=image_size,
        batch_size=batch_size,
        seed=123,
        validation_split=0.2,
        subset="validation",
    )



print(len(train_ds))
print(len(val_ds))

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

def make_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return tf.keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=7)

epochs = 50
verbose = 1

early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience / 4), verbose=1)

model_checkpoint = ModelCheckpoint("save_at_{epoch}.h5", 'val_loss', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, early_stop, reduce_lr]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)


