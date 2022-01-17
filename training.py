from tokenize import PlainToken
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import random

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

print(tf.__version__)
print(tf.config.list_physical_devices())
sys_details = tf.sysconfig.get_build_info()
cuda = sys_details["cuda_version"]
cudnn = sys_details["cudnn_version"]
print(cuda + ', ' + cudnn)

batch_size = 32
img_height, img_width = 256, 256

seed = 150

#Load training data
data_dir = 'train_data'
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Augment data to avoid overfitting training data
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model = keras.Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(32, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

'''
if os.path.exists('saved_model/model_1'):
  model = keras.models.load_model('saved_model/model_1')
  print('Model loaded')
'''

model.summary()

checkpoint_path = 'saved_models/model_1'
checkpoint_dir = os.path.dirname(checkpoint_path)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

keras.backend.clear_session()

'''
if os.path.exists(checkpoint_path + '.index'):
    latest = tf.train.latest_checkpoint(checkpoint_path)
    keras.backend.clear_session()
    model.load_weights(latest)
    print('Weights loaded')
'''

# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor='val_accuracy',
                                                save_weights_only=False,
                                                save_best_only=True,
                                                verbose=1)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                min_delta=0.01,
                                                patience=8,
                                                restore_best_weights=True)

epochs = 20

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[cp_callback, early_stopping]
)

# model.save(checkpoint_path)

#checkpoint_path = 'training_1/cp.ckpt'
#model.load_weights(checkpoint_path)