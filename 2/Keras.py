from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json
from keras.optimizers import adam, Adam
from keras.callbacks import ModelCheckpoint
import pandas
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

#   read model and weights
# model = model_from_json(open('my_model_architecture.json').read())
# model.load_weights('my_model_weights.h5')

#   build model
print("building model:")
model = Sequential()
model.add(Dense(300, input_shape=(11392, ),activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

#   batch train
print("batch training")
partnums = 200
width = 11392

for i in range(partnums):
    lines = open('train/part%04d' % i).readlines()

    print("reading the %04d part of train: " % i)

    depth = len(lines)

    print("depth:", depth)

    #   generate train_X and train_y
    train_X = [[0 for _ in range(width)] for __ in range(depth)]
    train_y = [0 for _ in range(depth)]

    for y, line in enumerate(lines):
        datas = line.split(' ')
        train_y[y] = int(datas[0])
        datas.pop(0)

        for data in datas:
            x = int(data.split(':')[0]) - 1
            train_X[y][x] = 1
            del x
        del datas

    print("width:", len(train_X[0]))

    model.fit(train_X, train_y, batch_size=50, nb_epoch=10, validation_split=0.05, callbacks=[checkpointer])

    del lines
    del train_X
    del train_y

#   save model and weights
json_string = model.to_json()
open('my_model_architecture.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5')