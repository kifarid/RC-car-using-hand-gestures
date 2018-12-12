import pickle as pkl
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, Model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import os
MODEL_PATH = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_PATH, 'hand_model_gray_INCEPTION.hdf5')
# path to the model weights files.
# dimensions of our images.
img_width, img_height = 170, 170

train_data_dir = 'training_data'
nb_train_samples = 1363
epochs = 1
batch_size = 64

# build the VGG16 network
top_model=applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',  input_shape=[img_width, img_height,3])
x = top_model.output
x=Flatten()(x)
x=Dense(64, activation='relu')(x)
x=Dense(32, activation='relu')(x)
x=Dropout(0.7)(x)
x=Dense(4, activation='softmax')(x)
model = Model(top_model.input,x)
print('Model loaded.')
#for layer in model.layers[:5]:
#   layer.trainable = True

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
# = optimizers.SGD(lr=1e-4, momentum=0.9)
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

training_generator = train_datagen.flow_from_directory(
    'training_data',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb',
)

# fine-tune the model
history=model.fit_generator(
    training_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs
    )
model.save(MODEL_FILE)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
with open('/history/inceptionV2.pkl', 'wb') as file_pi:
    pkl.dumps(history, file_pi)
model.save(MODEL_FILE)