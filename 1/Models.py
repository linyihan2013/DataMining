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

    # GradientBoostingRegressor

    # graBoos = GradientBoostingRegressor(learning_rate=0.3875)
    # graBoos.fit(x, y)

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