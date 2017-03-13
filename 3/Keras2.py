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
from scipy.ndimage import zoom

#   read model and weights
# model = model_from_json(open('my_model_architecture.json').read())
# model.load_weights('my_model_weights.h5')

#   build model
print("building model...")

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 128, 128), activation='relu'))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(9, activation='softmax'))
# sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="weights1.hdf5", verbose=1, save_best_only=True)

#   read images
lines = open('train/train.csv').readlines()
file = open('train/train.csv')
lines.pop(0)
file.readline()

depth = len(lines)
del lines

times = 2
each_part = int(depth / times)
others = depth % times

#   batch train
print("reading images...")

for time in range(times):
    print("time %d: " % time)
    lines = []
    if time < times:
        for line in range(each_part):
            lines.append(file.readline())

    image = numpy.empty((len(lines), 3, 128, 128),dtype="float32")
    label = numpy.empty((len(lines), ), dtype="uint8")
    left = numpy.empty((len(lines), ), dtype='str')

    for i, line in enumerate(lines):
        line = line.strip()
        input_img = Image.open("train/images/" + line.split(',')[0])

        # print("%d before: " % i)
        # print(list(input_img.getdata()))
        input_img = input_img.resize((128, 128), Image.ANTIALIAS)
        # print("%d after: " % i)
        # print(list(input_img.getdata()))
        # input_img.show()

        arr = numpy.asarray(input_img, dtype='float32') / 255
        image[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
        left[i] = line.split(',')[0]
        label[i] = int(line.split(',')[1])

        del input_img
        del arr

    label = to_categorical(label)
    image_train, image_test, label_train, label_test = train_test_split(image, label, test_size=0.1, random_state=1)

    model.fit(image_train, label_train, batch_size=25, nb_epoch=10, validation_data=(image_test, label_test), callbacks=[checkpointer])

    score = model.evaluate(image_test, label_test, verbose=0)
    print('Test score:', score)

    del lines
    del image
    del label
    del left
    del image_train
    del image_test
    del label_train
    del label_test

#   save model and weights
json_string = model.to_json()
open('my_model_architecture3.json', 'w').write(json_string)
model.save_weights('my_model_weights3.h5')