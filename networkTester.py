from distutils.log import error
import NeuralNetwork
from random import random

network = NeuralNetwork.Network(6, 3, 4, 4)

def feedCheck(atempts = 100):
    errors = 0
    for atempt in range(atempts):
        inputs = []
        for i in range(6):
            inputs.append(random())
        outs = network.feed(inputs)
        for i in range(len(outs)):
            if outs[i] > 1:
                print(f"error detected at: {atempt}:{i}, value: {outs[i]}")
                errors += 1
    print("test Complete")
    if errors == 0:
        print("without any errors")
    else:
        print(f"with {errors} errors")

feedCheck(10000)