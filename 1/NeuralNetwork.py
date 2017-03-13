import pandas
import matplotlib.pyplot
import seaborn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
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

    y = y.reshape(-1, 1)

    ds = SupervisedDataSet(x.shape[1], y.shape[1])
    ds.setField('input', x)
    ds.setField('target', y)

    hiddenSize = 100
    net = buildNetwork(x.shape[1], hiddenSize, 1, bias=True)
    net = pickle.load( open("model_val.pk1", 'rb' ))
    trainer = BackpropTrainer(net, ds)

    for i in range(300):
	    mse = trainer.train()
	    rmse = math.sqrt( mse )
	    print("training RMSE, epoch {}: {}".format( i + 1, rmse ))

    # trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=50, continueEpochs=10)

    pickle.dump( net, open("model_val.pk1", 'wb' ))

    y_test_dummy = numpy.zeros(y.shape)
    data2 = pandas.read_csv('test.csv', index_col=0)

    ds2 = SupervisedDataSet(data2.shape[1], y.shape[1])
    ds2.setField( 'input', data2)
    ds2.setField( 'target',y_test_dummy)

    y_pred = net.activateOnDataset(ds2)

    output = open("sub1.csv", "w")
    output.write("Id,reference\n")
    for i, result in enumerate(y_pred):
        output.write("%d,%.1f\n" % (i, result))
        print("%d,%.1f\n" % (i, result))
    output.close()