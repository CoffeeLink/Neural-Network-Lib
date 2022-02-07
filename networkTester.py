import NeuralNetwork
from random import random

network = NeuralNetwork.Network(4, 4, [6, 4, 4, 8])

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

print(network.feed([1, 2, 3, 4,]))

network.importNetwork("export.nns")

print(network.feed([0, 1]))
