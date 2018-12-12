import os # operating system functions (ie. path building on Windows vs. MacOs)

import cv2 # (OpenCV) computer vision functions (ie. tracking)
import matplotlib.pyplot as plt
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
MODEL_PATH = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_PATH, 'hand_model_gray_third.hdf5')
MODEL_HISTORY = os.path.join(MODEL_PATH, 'model_history.txt')
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 1), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 64
#optimizers.SGD(lr=0.0001, momentum=0.9)
training_datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.15,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

training_generator = training_datagen.flow_from_directory(
    'training_data',
    target_size=(100, 100),
    batch_size=batch_size,
    color_mode='grayscale',
    subset='training'
)

validation_generator = training_datagen.flow_from_directory(
    'training_data',
    target_size=(100, 100),
    batch_size=batch_size,
    color_mode='grayscale',
    subset='validation'
)

history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=21,
    epochs=100,
    validation_data= validation_generator,
    verbose=1,

)
model.save(MODEL_FILE)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()