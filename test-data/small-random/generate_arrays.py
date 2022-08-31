# The arrays in this folder were generated with this script

import numpy

numpy.random.seed(7)
test = numpy.random.rand(6, 1024).astype(numpy.float32)
train = numpy.random.rand(11, 1024).astype(numpy.float32)
numpy.save("test-data/test.npy", test)
numpy.save("test-data/train.npy", train)
