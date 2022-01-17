import os
from os import path

labels = ['chili-dog', 'hotdog', 'frankfurter', 'other']
train_path = 'train_data'
kaggle_path = 'train_kaggle'

def check_train_dir():
    if not path.exists(train_path):
        os.makedirs(train_path)

    for label in labels:
        label_path = path.join(train_path, label)
        if not path.exists(label_path):
            os.mkdir(label_path)

def sort_img(src):
    img = path.split(src)[1]
    for label in labels:
        if label == 'other' or img[:len(label)] == label:
            dst = path.join(train_path, label, img)
            if path.exists(dst):
                return

            try:
                os.rename(src, dst)
                return
            except OSError as error:
                pass

def convert_train_kaggle():
    if not path.exists(kaggle_path):
        print(kaggle_path + ' does not exist')
        return

    data_list = os.listdir(kaggle_path)
    for img in data_list:
        sort_img(path.join(kaggle_path, img))

def cull(fraction):
    for label in labels:
        label_path = path.join(train_path, label)
        img_list = os.listdir(label_path)
        num_remove = round(len(img_list) * fraction)
        for img in img_list[:num_remove]:
            os.remove(path.join(label_path, img))

check_train_dir()
convert_train_kaggle()
#cull(0.5)