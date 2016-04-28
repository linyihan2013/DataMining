import pandas
import matplotlib.pyplot
import seaborn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn import svm
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import numpy
import math
import pickle

if __name__ == '__main__':
    data = pandas.read_csv('train.csv', index_col=0)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    #LinearRgression

    # linreg = LinearRegression()
    # linreg.fit(x_train, y_train)
    # y_pred = linreg.predict(x_test)
    # print(numpy.sqrt(mean_squared_error(y_test, y_pred)))

    #BayesianRidge

    # ri = BayesianRidge()
    # ri.fit(x_train, y_train)
    # y_pred = ri.predict(x_test)
    # print(numpy.sqrt(mean_squared_error(y_test, y_pred)))

    #AdaBoostRegressor

    # graBoos = AdaBoostRegressor()
    # graBoos.fit(x_train, y_train)
    # y_pred = graBoos.predict(x_test)
    # print(numpy.sqrt(mean_squared_error(y_test, y_pred)))

    #SVM

    s = svm.SVR(kernel='rbf', C=5e3, gamma=0.1)
    s.fit(x_train, y_train)
    y_pred = s.predict(x_test)
    print(numpy.sqrt(mean_squared_error(y_test, y_pred)))

    #Neural Network

    # y_train = y_train.reshape(-1, 1)
    # y_test = y_test.reshape(-1, 1)

    # ds = SupervisedDataSet(x_train.shape[1], y_train.shape[1])
    # ds.setField('input', x_train)
    # ds.setField('target', y_train)
    #
    # hiddenSize = 100
    # net = buildNetwork(x.shape[1], hiddenSize, 1, bias=True)
    # trainer = BackpropTrainer(net, ds)

    # for i in range(100):
	 #    mse = trainer.train()
	 #    rmse = math.sqrt( mse )
	 #    print("training RMSE, epoch {}: {}".format( i + 1, rmse ))


    # trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=10, continueEpochs=10)
    #
    # pickle.dump( net, open("model.pk1", 'wb' ))
    #
    # y_test_dummy = numpy.zeros(y_test.shape)
    #
    # ds2 = SupervisedDataSet(x_train.shape[1], y_test.shape[1])
    # ds2.setField( 'input',x_test)
    # ds2.setField( 'target',y_test_dummy)
    #
    # y_pred = net.activateOnDataset(ds2)
    # print(numpy.sqrt(mean_squared_error(y_test, y_pred)))

    #Pyplot

    # seaborn.pairplot(data, x_vars=['value0', 'value1', 'value2', 'value3'], y_vars='reference', size=7, aspect=0.7)
    # matplotlib.pyplot.show()