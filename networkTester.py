import NeuralNetwork
from random import random

network = NeuralNetwork.Network(9, 4, [10, 10], [1, 1, 1])
network.clearPatterns()

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


network.addPattern([1,0,1,0,1,0,1,0,1], [1, 0, 0, 0])# x
network.addPattern([0,1,0,1,0,1,0,1,0], [0, 1, 0, 0])# +
network.addPattern([0,0,0,1,1,1,0,0,0], [0, 0, 1, 0])# -
network.addPattern([0,0,1,0,1,0,1,0,0], [0, 0, 0, 1])# /

network.backpropogate(1200, 0.3)

print(f"#-#\n-#-\n#-#\n {network.feed([1,0,1,0,1,0,1,0,1])}")