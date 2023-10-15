import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.applications import NASNetMobile, InceptionV3
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


#################################### DATA ###################################

batch_size = 64
img_height = 224
img_width = 224


train_ds_path = "E:/data/Drinking_4class/Train/"
val_ds_path = "E:/data/Drinking_4class/Val/"

# train_ds_path = "E:/data/UEH_vending/UEH_vending_train/"
# val_ds_path = "E:/data/UEH_vending/UEH_vending_val/"

train_ds_path = "E:/data/kaggle_trashbox/"
val_ds_path = "E:/data/Recycling/Recycling_test/"

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_ds_path,
  image_size=(img_height, img_width),
  interpolation='area',
  label_mode = 'categorical',
  shuffle=True,
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
#   "E:/whiten_split_224/val/",
  val_ds_path,  
  image_size=(img_height, img_width),
  interpolation='area',
  label_mode = 'categorical', 
  shuffle=False,
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(5509).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#################################### Augmentation layer ######################

aug_Flip_vert = layers.RandomFlip("vertical")

aug_Flip_hor = layers.RandomFlip("horizontal")

aug_Crop = keras.Sequential(
    [
    layers.RandomCrop(height = 180, width = 180),
    layers.Resizing(height = 224, width = 224, interpolation="area")
    ]
)
aug_Rot = layers.RandomRotation(factor = 0.25)

aug_Contrast = layers.RandomContrast(factor = 0.5)
#################################### MODEL ###################################
def Create_model(model_name, augment):
  num_classes = len(class_names)
  n_class = num_classes
  conv_base = model_name(
                    include_top=False,
                    input_shape=(img_width, img_height, 3),
                    pooling="avg"
                    )  # 3 = number of channels in RGB pictures
  conv_base.trainable = False

  model = Sequential()
  model.add(tf.keras.Input(shape=(img_width, img_height, 3)))
  model.add(layers.Rescaling(1./255))
  model.add(augment)
  model.add(conv_base)

  model.add(layers.Dropout(0.8))
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.8))
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dropout(0.7))
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dense(n_class, activation='softmax'))


  model.compile(optimizer="adam",
                loss='categorical_crossentropy',
                metrics=['acc'])
  
  return model



def Train_save(model, epochs, name):

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=True
  )



  model_name = name
  model.save("./trained_model/" + model_name +".h5")

  # Get the dictionary containing each metric and the loss for each epoch
  history_dict = history.history
  # Save it under the form of a json file
  json.dump(history_dict, open('./json/'+ model_name + '.json', 'w'))

epochs = 30

model_Inc_flip_vert = Create_model(InceptionV3, aug_Flip_vert)
model_Inc_flip_hor = Create_model(InceptionV3, aug_Flip_hor)
model_Inc_Crop = Create_model(InceptionV3, aug_Crop)
model_Inc_Rot = Create_model(InceptionV3, aug_Rot)
model_Inc_Contrast = Create_model(InceptionV3, aug_Contrast)

Train_save(model_Inc_flip_vert, epochs, "model_Inc_flip_vert")
Train_save(model_Inc_flip_hor, epochs, "model_Inc_flip_hor")
Train_save(model_Inc_Crop, epochs, "model_Inc_Crop")
Train_save(model_Inc_Rot, epochs, "model_Inc_Rot")
Train_save(model_Inc_Contrast, epochs, "model_Inc_Contrast")