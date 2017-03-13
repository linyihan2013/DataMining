from sklearn import svm
import numpy

if __name__ == '__main__':
    # partnum = 0
    #
    # #   train
    # lines = open('part%04d'%partnum).readlines()
    #
    # depth = len(lines)
    # width = 11392
    #
    # print("depth:", depth)
    #
    # train_X = [[0 for _ in range(width)] for __ in range(depth)]
    # train_y = [0 for _ in range(depth)]
    #
    # for y, line in enumerate(lines):
    #     datas = line.split(' ')
    #     train_y[y] = int(datas[0])
    #     datas.pop(0)
    #
    #     for data in datas:
    #         x = int(data.split(':')[0]) - 1
    #         train_X[y][x] = 1
    #
    # print("width:", len(train_X[0]))
    #
    # clf = svm.LinearSVC()
    #
    # clf.fit(train_X, train_y)
    # #   test
    #
    # lines = open("test.txt").readlines()
    #
    # predicts = [[0 for _ in range(width)] for __ in range(len(lines))]
    #
    # for y, line in enumerate(lines):
    #     datas = line.split(' ')
    #     datas.pop(0)
    #
    #     for data in datas:
    #         x = int(data.split(':')[0]) - 1
    #         predicts[y][x] = 1
    # results = clf.predict(predicts)

    lines = open("input.csv").readlines()
    print("depth:", len(lines))

    output = open("submission.csv", "w")
    output.write("Id,label\n")

    for i, result in enumerate(lines):
        output.write("%d,%s" % (i, result))
        print("%d,%s" % (i, result))
    output.close()