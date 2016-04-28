import pandas
import matplotlib.pyplot
import seaborn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import neural_network
import numpy

if __name__ == '__main__':
    data = pandas.read_csv('train.csv', index_col=0)
    cols = data.columns
    y = data.iloc[:, -1]
    for i in cols:
        x = data.iloc[:, :-1]
        del x[i]
        print(x.shape)
        # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
        # linreg = LinearRegression()
        # linreg.fit(x_train, y_train)
        # y_pred = linreg.predict(x_test)
        # print(numpy.sqrt(mean_squared_error(y_test, y_pred)))