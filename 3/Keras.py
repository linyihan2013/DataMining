from keras.models import Sequential
from keras.layers.core import Dense, Dropout, RepeatVector, Merge
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.models import model_from_json
from keras.optimizers import adam, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import pandas
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from PIL import Image

#   read model and weights
# model = model_from_json(open('my_model_architecture.json').read())
# model.load_weights('my_model_weights.h5')

#   build model
print("building model...")
image1_model = Sequential()
image1_model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(3, 20, 20), activation='relu'))
image1_model.add(Convolution2D(64, 3, 3, activation='relu'))
image1_model.add(MaxPooling2D(pool_size=(2, 2)))

image1_model.add(Flatten())
image1_model.add(Dense(20, activation='relu'))

image2_model = Sequential()
image2_model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(3, 20, 20), activation='relu'))
image2_model.add(Convolution2D(64, 3, 3, activation='relu'))
image2_model.add(MaxPooling2D(pool_size=(2, 2)))

image2_model.add(Flatten())
image2_model.add(Dense(20, activation='relu'))

image3_model = Sequential()
image3_model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(3, 20, 20), activation='relu'))
image3_model.add(Convolution2D(64, 3, 3, activation='relu'))
image3_model.add(MaxPooling2D(pool_size=(2, 2)))

image3_model.add(Flatten())
image3_model.add(Dense(20, activation='relu'))

image4_model = Sequential()
image4_model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(3, 20, 20), activation='relu'))
image4_model.add(Convolution2D(64, 3, 3, activation='relu'))
image4_model.add(MaxPooling2D(pool_size=(2, 2)))

image4_model.add(Flatten())
image4_model.add(Dense(20, activation='relu'))

model = Sequential()
model.add(Merge([image1_model, image2_model, image3_model, image4_model], mode='concat'))
model.add(Dense(9, activation='softmax'))
# sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="weights1.hdf5", verbose=1, save_best_only=True)

#   read images
lines = open('train/train.csv').readlines()
lines.pop(0)

#   batch train
print("reading images...")
image = numpy.empty((len(lines), 3, 64, 64),dtype="float32")
label = numpy.empty((len(lines), ), dtype="uint8")
left = numpy.empty((len(lines), ), dtype='str')

for i, line in enumerate(lines):
    line = line.strip()
    input_img = Image.open("train/images/" + line.split(',')[0])
    arr = numpy.asarray(input_img, dtype='float32') / 255
    image[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
    left[i] = line.split(',')[0]
    label[i] = int(line.split(',')[1])

label = to_categorical(label)
image_train, image_test, label_train, label_test = train_test_split(image, label, test_size=0.1, random_state=1)
data1_train = numpy.empty((len(image_train), 3, 20, 20), dtype="float32")
data1_test = numpy.empty((len(image_test), 3, 20, 20), dtype="float32")
data2_train = numpy.empty((len(image_train), 3, 20, 20), dtype="float32")
data2_test = numpy.empty((len(image_test), 3, 20, 20), dtype="float32")
data3_train = numpy.empty((len(image_train), 3, 20, 20), dtype="float32")
data3_test = numpy.empty((len(image_test), 3, 20, 20), dtype="float32")
data4_train = numpy.empty((len(image_train), 3, 20, 20), dtype="float32")
data4_test = numpy.empty((len(image_test), 3, 20, 20), dtype="float32")

for i in range(len(image_train)):
    for x in range(20):
        for y in range(20):
            data1_train[i, :, x, y] = image_train[i, :, x, 20 + y]
            data2_train[i, :, x, y] = image_train[i, :, x, y]
            data3_train[i, :, x, y] = image_train[i, :, 20 + x, 20 + y]
            data4_train[i, :, x, y] = image_train[i, :, 40 + x, 20 + y]

for i in range(len(image_test)):
    for x in range(20):
        for y in range(20):
            data1_test[i, :, x, y] = image_test[i, :, x, 20 + y]
            data2_test[i, :, x, y] = image_test[i, :, x, y]
            data3_test[i, :, x, y] = image_test[i, :, 20 + x, 20 + y]
            data4_test[i, :, x, y] = image_test[i, :, 40 + x, 20 + y]

model.fit([data1_train, data2_train, data3_train, data4_train], label_train, batch_size=50, nb_epoch=100, validation_data=([data1_test, data2_test, data3_test, data4_test], label_test), callbacks=[checkpointer])

score = model.evaluate([data1_test, data2_test, data3_test, data4_test], label_test, verbose=0)
print('Test score:', score)

#   save model and weights
json_string = model.to_json()
open('my_model_architecture3.json', 'w').write(json_string)
model.save_weights('my_model_weights3.h5')