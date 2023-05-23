#modelimport

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense ,Flatten ,Conv2D ,MaxPooling2D ,Dropout ,BatchNormalization
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau , ModelCheckpoint
from keras.applications.mobilenet import MobileNet ,preprocess_input
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.config.run_functions_eagerly(True)
optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.99)

def mobilenet_model(training, testing, validing):
    mobilenet=MobileNet(include_top=False,weights='imagenet',input_shape=(224,224,3))
    mobilenet.trainable=False
    mob_model=Sequential([
        mobilenet,
        MaxPooling2D(3,2),
        Flatten(),
        Dense(128,activation='relu'),
        BatchNormalization(),
        Dense(1024,activation='relu'),
        BatchNormalization(),
        Dense(512,activation='relu'),
        BatchNormalization(),
        Dense(132,activation='softmax')
    ])
    mob_model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=["accuracy", "Precision", "Recall", "AUC"])
    epochs = 10
    batch_size=32
    steps_per_epoch = training.n // batch_size
    validation_steps = validing.n // batch_size
    history_mob=mob_model.fit(training,validation_data=validing,epochs=50,batch_size=batch_size,
                          steps_per_epoch=steps_per_epoch,validation_steps=validation_steps, verbose=0)
    return history_mob
