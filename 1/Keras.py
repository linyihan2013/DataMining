from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json
from keras.optimizers import adam, Adam
from keras.callbacks import ModelCheckpoint
import pandas
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

data = pandas.read_csv('train.csv', index_col=0)
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

#
# model = model_from_json(open('my_model_architecture2.json').read())
# model.load_weights('my_model_weights2.h5')

model = Sequential()
model.add(Dense(300, input_shape=x_train.shape[1:],activation='relu'))
model.add(Dense(300, activation='sigmoid'))
model.add(Dense(300, activation='tanh'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint(filepath="weights1.hdf5", verbose=1, save_best_only=True)
model.fit(x, y, batch_size=25, nb_epoch=300, validation_split=0.05, callbacks=[checkpointer])

predicted = model.predict(x_test)
print("RMSE:", numpy.sqrt(mean_squared_error(y_test, predicted)))

json_string = model.to_json()
open('my_model_architecture2.json', 'w').write(json_string)
model.save_weights('my_model_weights2.h5')