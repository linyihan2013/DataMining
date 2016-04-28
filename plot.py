import pandas
import matplotlib.pyplot
import seaborn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy

if __name__ == '__main__':
    data = pandas.read_csv('train.csv', index_col=0)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    seaborn.pairplot(data, x_vars=['value0', 'value1', 'value2', 'value3', 'value4', 'value5', 'value6', 'value7'], y_vars='reference', size=7, aspect=0.7)
    matplotlib.pyplot.show()