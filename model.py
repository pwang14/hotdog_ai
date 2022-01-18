from tokenize import PlainToken
import os
from os import path
import PIL

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

'''
print(tf.__version__)
print(tf.config.list_physical_devices())
sys_details = tf.sysconfig.get_build_info()
cuda = sys_details["cuda_version"]
cudnn = sys_details["cudnn_version"]
print(cuda + ', ' + cudnn)
'''

print()

UPLOAD_FOLDER = os.path.join('static', 'temp_files')
img_height, img_width = 256, 256

model_path = 'saved_models/model_1'
if path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print('Model loaded')
else:
    model = None

def apply(path):
    if not model:
        return -1

    img = tf.keras.utils.load_img(path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return 1 - np.argmax(score)

