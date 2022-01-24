import logging
import math
import numpy as np
from random import randint, random

#funkciok

def sigmoid(x): #sigmoid func 
  return 1 / (1 + math.exp(-x))

#föcuc

class layer:
    neuronCount : int # neuron num
    weightMatrix : list # [[(1,1), (1,2), (1,n) ++], [(2,1), (2,2), (2,n) ++], [(n,1), (n,2), (n,n) ++] ++] matrix moment
    #neuronValueVector : list # [(1,Neuron value), (2,Neuron value), (n,Neuron value) ++]
    biasVector : list # [(1,bias value), (2,bias value), (n,bias value) ++]
    def __init__(self, neuronCount, weights, biases):
        self.neuronCount = neuronCount
        self.weightMatrix = weights
        self.biasVector = biases

    def calculateNeurons(self, perviousLayerNeurons : list):
        neuronVector = []
        logging.debug(f" neuron calc start: {perviousLayerNeurons}, {self.biasVector}, {self.weightMatrix}")
        for b, w in zip(self.biasVector, self.weightMatrix): # b, w. b = biases. w = weight
            logging.debug(f"b: {b}, w: {w}")
            neuron = sigmoid(np.dot(w, perviousLayerNeurons) + b) # meg szorozuk a neuront a weightekkel és hozáadjuk a biast és utána sigmoid moment
            logging.debug(f"neuron out: {neuron}")
            neuronVector.append(neuron)
            logging.debug(f"total {neuronVector}")
        return neuronVector


class Network:
    hiddenLayers : list
    outLayer : layer
    layers : int
    def __init__(self, inputNeurons, outputNeurons, layers = 2, neuronsPerLayer = 10, baseNetwork = None, debug=False):
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # variables
        self.layers = layers
        self.hiddenLayers = []
        # L1 weight matrix generation
        weightMatrix = []
        for i1 in range(neuronsPerLayer):
            weights = []
            for i2 in range(inputNeurons):
                weights.append(randint(-1, 1))
            weightMatrix.append(weights)

        biases = []
        for i in range(neuronsPerLayer):
            biases.append(0)

        gLayer = layer(neuronsPerLayer, weightMatrix, biases)
        self.hiddenLayers.append(gLayer)

        # auto gen layers
        for i in range(layers - 1):
            weightMatrix = []
            for i1 in range(neuronsPerLayer):
                weights = []
                for i2 in range(neuronsPerLayer):
                    weights.append(randint(-1, 1))
                weightMatrix.append(weights)
            biases = []
            for i in range(neuronsPerLayer):
                biases.append(0)

        gLayer = layer(neuronsPerLayer, weightMatrix, biases)
        self.hiddenLayers.append(gLayer)

        # final out layer
        weightMatrix = []
        for i1 in range(outputNeurons):
            weights = []
            for i2 in range(neuronsPerLayer):
                weights.append(randint(-1, 1))
            weightMatrix.append(weights)

        biases = []
        for i in range(neuronsPerLayer):
            biases.append(0)
        
        outLayer = layer(outputNeurons, weightMatrix, biases)
        self.outLayer = outLayer
        logging.info("Network Generation Complete!")
    
    def feed(self, dataVector):
        """'feeds the network'. röviden tömören be adod az adatot
         azaz az input layert és visza adja a válasz vektort
         nem választ hogy melyik a main"""
        for i in range(len(self.hiddenLayers)):
            dataVector = self.hiddenLayers[i].calculateNeurons(dataVector)
        return self.outLayer.calculateNeurons(dataVector)

