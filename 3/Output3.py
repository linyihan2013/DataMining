from keras.models import model_from_json
from PIL import Image
import numpy
import sys
from keras.layers.core import Layer
import theano.tensor as T

class LRN(Layer):

    def __init__(self, alpha=0.0001,k=1,beta=0.75,n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2 # half the local region
        input_sqr = T.sqr(x) # square the input
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c) # make an empty tensor with zero pads along channel dimension
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr) # set the center to be the squared input
        scale = self.k # offset for the scale
        norm_alpha = self.alpha / self.n # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolHelper(Layer):

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:,:,1:,1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#   read model and weights
model = model_from_json(open('googlenet_architecture.json').read(),custom_objects={"PoolHelper": PoolHelper,"LRN":LRN})
model.load_weights('my_model_weights2.h5')
# model.load_weights('weights.hdf5')

#   read images
lines = open('test/test.csv').readlines()
lines.pop(0)

#   predict
print("reading images...")
output = open("submission5.csv", "w")
output.write("Id,label\n")

image = numpy.empty((len(lines), 3, 224, 224),dtype="float32")
label = numpy.empty((len(lines), ), dtype="uint8")

for i, line in enumerate(lines):
        line = line.strip()
        input_img = Image.open("test/images/" + line).resize((256, 256), Image.ANTIALIAS)
        arr = numpy.asarray(input_img, dtype='float32')
        height,width = arr.shape[:2]
        arr[:, :, 0] -= 123.68
        arr[:, :, 1] -= 116.779
        arr[:, :, 2] -= 103.939
        arr[:,:,[0,1,2]] = arr[:,:,[2,1,0]] # swap channels
        arr = arr.transpose((2, 0, 1)) # re-order dimensions
        arr = arr[:,(height-224)//2:(height+224)//2,(width-224)//2:(width+224)//2] #crop
        arr = numpy.expand_dims(arr, axis=0) # add dimension for batch

        image[i,:,:,:] = arr[:,:,:]

        del input_img
        del arr

print("predicting...")

predicted = model.predict(image)
print(type(predicted[2][0]))
print(len(predicted[2][0]))

print("outputing...")

for i, line in enumerate(lines):
    line = line.strip()
    output.write("%s,%s\n" % (line, numpy.argmax(predicted[2][i])))

output.close()