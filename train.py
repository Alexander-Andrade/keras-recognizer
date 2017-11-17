import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

def n_samples(path):
    return len(glob.glob(path +'/*'))

batch_size = 4
epochs = 15

train_data_dir = 'train_google_earth/train'
validation_data_dir = 'train_google_earth/validation'

img_width, img_height = 300, 300

n_has_buildings = n_samples('train_google_earth/train/has_buildings')
n_has_not_buildings = n_samples('train_google_earth/train/has_no_buildings')

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

n_train_samples = train_generator.samples

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

n_validation_samples = validation_generator.samples

model.fit_generator(
    train_generator,
    steps_per_epoch=n_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=n_validation_samples // batch_size)

model.save_weights('trained_google_earth.h5')
