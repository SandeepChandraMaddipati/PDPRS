#Dataloader

import tensorflow as tf
from keras.applications.mobilenet import MobileNet ,preprocess_input

path = r"D:\PDARS\version2.1.9\new_dataset - Copy\DATASET"

def dataloader(path):
    trainer=tf.keras.preprocessing.image.ImageDataGenerator(
        zca_epsilon=1e-06,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest",
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input,
        validation_split=0.5
    ).flow_from_directory(path,batch_size=16,target_size=(224,224),subset="training")

    valider=tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest",
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input,
        validation_split=0.5
    ).flow_from_directory(path,batch_size=16,target_size=(224,224),subset='validation',shuffle=False)

    tester=tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
    ).flow_from_directory(path,batch_size=16,target_size=(224,224),shuffle=False)

    return trainer, valider, tester
