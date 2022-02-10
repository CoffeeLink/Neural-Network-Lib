import NeuralNetwork
from random import random

network = NeuralNetwork.Network(2, 2, [4, ], [1, 1, ])

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

#network.importNetwork("save01.nns")

network.clearPatterns()

network.addPattern([0, 1], [1, 0])
network.addPattern([1, 0], [1, 0])
network.addPattern([1, 1], [1, 0])
network.addPattern([0, 0], [0, 1])


network.backpropogate(20000, 0.45)

print(network.feed(inputVector=[1, 0]))
print(network.feed(inputVector=[0, 1]))

print(network.feed(inputVector=[1, 1]))
print(network.feed(inputVector=[0, 0]))

network.exportNetwork("save01.nns")

