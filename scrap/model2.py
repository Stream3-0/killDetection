import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pathlib
import cv2
import glob


train_dataset = tf.keras.utils.image_dataset_from_directory(
    './training_data',
    validation_split = 0.2,
    subset = "training",
    seed = 24,  
    image_size = (227, 227),
    batch_size = 32
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    './training_data',
    validation_split = 0.2,
    subset = "validation",
    seed = 24,  
    image_size = (227, 227),
    batch_size = 32
)

train_dataset = train_dataset.shuffle(13).cache().prefetch(buffer_size = tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(13).cache().prefetch(buffer_size = tf.data.AUTOTUNE)

model = Sequential ([
    layers.Rescaling(1./255, input_shape = (227, 227, 3)),
    layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(2)
])

model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

model.fit (
    train_dataset, 
    validation_data = val_dataset,
    epochs = 20
)
model.save('model.h5')