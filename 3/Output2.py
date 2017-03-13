from keras.models import model_from_json
from PIL import Image
import numpy
import sys

#   read model and weights
model = model_from_json(open('googlenet_architecture.json').read())
model.load_weights("weights1.hdf5")
# model.load_weights('weights.hdf5')

#   read images
lines = open('test/test.csv').readlines()
lines.pop(0)

#   predict
print("reading images...")
output = open("submission.csv", "w")
weights = open("weights.csv", "w")
output.write("Id,label\n")
weights.write("Id,weight\n")

data1 = numpy.empty((len(lines), 3, 20, 20), dtype="float32")
data2 = numpy.empty((len(lines), 3, 20, 20), dtype="float32")
data3 = numpy.empty((len(lines), 3, 20, 20), dtype="float32")
image = numpy.empty((3, 64, 64),dtype="float32")
left = numpy.empty((len(lines), ), dtype='str')

for i, line in enumerate(lines):
    line = line.strip()
    input_img = Image.open("test/images/" + line)
    arr = numpy.asarray(input_img, dtype='float32') / 255
    image[:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
    for x in range(20):
        for y in range(20):
            data1[i, :, x, y] = image[:, x, 20 + y]
            data2[i, :, x, y] = image[:, x, y]
            data3[i, :, x, y] = image[:, 20 + x, 20 + y]
    left[i] = line

print("predicting...")

predicted = model.predict([data1, data2, data3])
print(predicted)

print("outputing...")

for i, line in enumerate(lines):
    line = line.strip()
    output.write("%s,%d\n" % (line, numpy.argmax(predicted[i])))
    weights.write(str(predicted[i]))

output.close()
weights.close()