from keras.models import model_from_json

#   read model and weights
model = model_from_json(open('my_model_architecture.json').read())
# model.load_weights('my_model_weights.h5')
model.load_weights('weights.hdf5')

#   predict
print("predicting:")
output = open("submission.csv", "w")
output.write("Id,label\n")
width = 11392

for i in range(20):
    print("reading part %04d of test" % i)
    lines = open("test/part%04d" % i).readlines()
    x_test = [[0 for _ in range(width)] for __ in range(len(lines))]
    indexs = []

    for y, line in enumerate(lines):
        datas = line.split(' ')
        indexs.append(datas[0])
        datas.pop(0)

        for data in datas:
            x = int(data.split(':')[0]) - 1
            x_test[y][x] = 1
            del x
        del datas
    predicted = model.predict(x_test)
    for j in range(len(indexs)):
        if float(predicted[j]) >= 0.5:
            output.write("%s,%d\n" % (indexs[j], 1))
        else:
            output.write("%s,%d\n" % (indexs[j], 0))

    del lines
    del indexs
    del predicted

output.close()