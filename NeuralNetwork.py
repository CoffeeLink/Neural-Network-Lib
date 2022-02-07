import logging
import math
import numpy as np
from random import randint
import json

#funkciok

def sigmoid(x): #sigmoid func 
  return 1 / (1 + math.exp(-x))

#main class

class layer:
    """A layer class that is reponsible for handling the value of the neurons in this layer, has 2 functions '__init__' and 'calculateNeurons' each have their part of the network"""
    neuronCount : int # neuron num
    weightMatrix : list # [[(1,1), (1,2), (1,n) ++], [(2,1), (2,2), (2,n) ++], [(n,1), (n,2), (n,n) ++] ++] matrix moment
    #neuronValueVector : list # [(1,Neuron value), (2,Neuron value), (n,Neuron value) ++]
    biasVector : list # [(1,bias value), (2,bias value), (n,bias value) ++]
    def __init__(self, neuronCount, weights, biases):
        """initalizes the layer just sets some values

        Args:
            neuronCount (int): sets the ammount of neurons in the layer
            weights (list): a matrix that contains the weight information of the layer
            biases (list): a 'biasVector' basically a list of biases for each neuron
        """
        self.neuronCount = neuronCount
        self.weightMatrix = weights
        self.biasVector = biases

    def calculateNeurons(self, perviousLayerNeurons : list):
        """calculate the next data vector

        Args:
            perviousLayerNeurons (list): the data vector of the perious layer

        Returns:
            [list]: the final calculated data vector for the next layer/output
        """
        neuronVector = []
        for b, w in zip(self.biasVector, self.weightMatrix): # b, w. b = biases. w = weight
            neuron = sigmoid(np.dot(w, perviousLayerNeurons) + b) # meg szorozuk a neuront a weightekkel és hozáadjuk a biast és utána sigmoid moment
            neuronVector.append(neuron)
        return neuronVector


class Network:
    Layers : list
    neuronCount : list
    activationType = 1 # gonna make it changeble some time later.
    def __init__(self, inputNeurons, outputNeurons, neuronsPerLayer : list, debug=False):
        """Initalizes the network and generates a network based of the given information.

        Args:
            inputNeurons (int): the ammount of neurons for the input layer. based of hov many data points you want to give to the network.
            outputNeurons (int): the ammount of neuron for the output layer. based of the ammount of data you want the network to return/choose from.
            neuronsPerLayer (list): a list of the ammount of neurons/hidden layer. NOTE: The length of the list is the number of layers you have(Out layer does not count)
            debug (bool, optional): makes it so it logs more data. Used for debugging the network, I HIGHLY reccomend using it if you make modifications for the code / your network isn't performing well. Defaults to False.
        """
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        logging.info("Network Initalizing...")

        # variables
        self.Layers = []
        self.neuronCount = []
        for item in neuronsPerLayer:
            self.neuronCount.append(item)
        self.neuronCount.append(outputNeurons)
        # Generation
        perv = inputNeurons
        for item in self.neuronCount:
            self.Layers.append(self.generateLayer(item, perv))
            perv = item

        logging.info("Generation Complete!") # generation end

    def feed(self, dataVector : list):
        """Calculates the output of the network

        Args:
            dataVector (list): the input data of the network

        Returns:
            list: the output of the network 
        """
        for item in self.Layers:
            dataVector = item.calculateNeurons(dataVector)
        return dataVector

    def exportNetwork(self, name):
        """Saves the network in a file

        Args:
            name (str): The save location for instace: /json/save.nns
        """
        with open(name, "w+") as f:
            f.truncate(0)
            parsed = {"layers": []}
            for item in self.Layers:
                parsed["layers"].append({"weights": item.weightMatrix, "biases": item.biasVector, "isInputLayer": False, "activationType": 1})
            f.write(json.dumps(parsed, indent=4, sort_keys=True))

    def importNetwork(self, name):
        """loads a saved network from file

        Args:
            name (str): the location of the save to import. for instance: /json/save.nns
        """
        with open(name, "r") as f:
            parsed = json.load(f)
            layers = []
            for item in parsed["layers"]:
                if item["isInputLayer"] != True:
                    layers.append(self.generateLayer(None, None, item)) #none is just a place holder cuz it dosent need any value but still it needs something
            self.Layers = layers

    def generateLayer(self, neurons, perviousNeuronCount, values=None):
        """Generates A layer Used By the main Network class

        Args:
            neurons (int): number of neurons 
            perviousNeuronCount (int): pervious number of neurons
            values (Dict, optional): used by the network only. Defaults to None.

        Returns:
            [layer]: the generated layer
        """
        if values == None:
            weightMatrix = []
            for i in range(neurons): # ahány neuron ja lesz ennek a layernek aanyi szor lesz egy kis vektorja
                weights = []
                for i2 in range(perviousNeuronCount): # elözö layer neuronjai vector
                    weights.append(randint(-1, 1))
                weightMatrix.append(weights)
            biases = [] # bias vector generácio itt keszdödik
            for i in range(neurons):
                biases.append(0)
            return layer(neurons, weightMatrix, biases)
        else:
            return layer(len(values["weights"]), values["weights"], values["biases"])

