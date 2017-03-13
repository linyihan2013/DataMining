from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json
from keras.optimizers import adam, Adam
from keras.callbacks import ModelCheckpoint
import pandas
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

data = pandas.read_csv('train.csv', index_col=0)
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

predicted = model.predict(x_test)
print("RMSE:", numpy.sqrt(mean_squared_error(y_test, predicted)))

data2 = pandas.read_csv('test.csv', index_col=0).values
results = model.predict(data2)

output = open("sub.csv", "w")
output.write("Id,reference\n")
for i, result in enumerate(results):
    output.write("%d,%.1f\n" % (i, result))
    # print("%d,%.1f\n" % (i, result))
output.close()