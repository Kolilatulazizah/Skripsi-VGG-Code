import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
## Special for CNN
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras_vggface.vggface import VGGFace
from keras.engine.base_layer import Layer
from tensorflow.keras.models import Model
import json

# Set environment variables to disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
folder = "model_kelas50"
try:
    os.mkdir(folder)
except:
    pass

train = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                           brightness_range=(0.8, 1.2),
                           zoom_range=[0.8, 1.2],
                           rotation_range=27)
validation = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

# load dataset
train_dataset = train.flow_from_directory(r'E:\PERKULIAHAN SEMESTER 7\Skripsi\Kolilatul Azizah\DATASET VGG\Data Wajah 50 kelas\training', target_size=(224,224), batch_size = 8, class_mode = 'categorical', color_mode="rgb")
validation_dataset = validation.flow_from_directory(r'E:\PERKULIAHAN SEMESTER 7\Skripsi\Kolilatul Azizah\DATASET VGG\Data Wajah 50 kelas\validation',target_size= (224, 224), batch_size = 8, class_mode = 'categorical', color_mode="rgb")
print(train_dataset.class_indices)

json_data = [i.strip().lower().title() for i in train_dataset.class_indices]
with open(folder+"\\"+folder+".json", "w") as file:
    json.dump(json_data, file)
class_count = len(train_dataset.class_indices)



base_model = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3))

#base_model.layers[0].trainable = False

from keras.models import Sequential
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout

x = base_model.output
x = Flatten()(x)

x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# final layer with softmax activation
preds = Dense(class_count, activation='softmax')(x)

# create a new model with the base model's original input and the
# new model's output
model = Model(inputs = base_model.input, outputs = preds)

# don't train the first 19 layers - 0..18
for layer in model.layers[:19]:
    layer.trainable = False

# train the rest of the layers - 19 onwards
for layer in model.layers[19:]:
    layer.trainable = True


#setting compiler
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])
print(model.summary())
input("....")


simpan = folder+'\\HasilTrainingvgg_layertrainablefalse_Epoch_{epoch:02d}.h5'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=simpan, monitor='val_loss', mode='min', verbose=1, save_best_only=True)


# paling baik saat loss sekitar 0.04
model_fit = model.fit(train_dataset, epochs=10, validation_data=validation_dataset, callbacks=[model_checkpoint])




plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(model_fit.history['loss'], label='Training Loss')
plt.plot(model_fit.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(model_fit.history['accuracy'], label='Training Accuracy')
plt.plot(model_fit.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.savefig(folder+"\\Hasil.png")




#model.save(r'C:\Users\Asus\.atom\CNN Artificial Intelligence\HasilTraining.h5')
