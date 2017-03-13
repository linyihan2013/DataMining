from sklearn import ensemble
import numpy

if __name__ == '__main__':
    lines = open("train.csv").readlines()
    lines.pop(0)
    length = len(lines)
    clf = ensemble.GradientBoostingRegressor()
    args = []
    results = []
    for line in lines:
        elements = numpy.array(line.split(',')).astype(numpy.double)
        args.append(elements[1:-1])
        results.append(elements[-1])
    clf.fit(args, results)

    lines = open("test.csv").readlines()
    lines.pop(0)
    predicts = []
    for line in lines:
        elements = numpy.array(line.split(',')).astype(numpy.double)
        predicts.append(elements[1:])
    results = clf.predict(predicts)

    output = open("submission.csv", "w")
    output.write("Id,reference\n")
    for i, result in enumerate(results):
        output.write("%d,%.1f\n" % (i, result))
        print("%d,%.1f\n" % (i, result))
    output.close()

    # lines = open("train.csv").readlines()
    # for line in lines:
    #     print(line)

    # lines = open("sample_submission.csv").readlines()
    # for line in lines:
    #     print(line)