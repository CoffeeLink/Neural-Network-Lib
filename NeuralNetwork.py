import logging
import math
import numpy as np
from random import random, choice
import json


# static acces vars
LINEAR = 0 # activation modell id so u can acces it by Network.LINEAR/SIGMOID/TANH/RELU
SIGMOID = 1
TANH = 2
RELU = 3

#funkciok
class relu:
    def relu(self, x):
        return max(0.0, x)
    def derivative(self, x):
        if x >= 0:
            return 1
        else:
            return 0

class linear:
    def linear(self, x):
        return x
    def derivative(self, x):
        return 1

class tanh:
    def tanh(self, x):
        return math.tanh(x)
    def derivative(self, x):
        return 1 - math.Pow(math.tanh(x), 2)
    
class sigmoid:
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    def derivative(self, x):
        return x * (1 - x)


#main class

class layer:
    """A layer class that is reponsible for handling the value of the neurons in this layer, has 2 functions '__init__' and 'calculateNeurons' each have their part of the network"""
    neuronCount : int # neuron num
    weightMatrix : list # [[(1,1), (1,2), (1,n) ++], [(2,1), (2,2), (2,n) ++], [(n,1), (n,2), (n,n) ++] ++] matrix moment
    #neuronValueVector : list # [(1,Neuron value), (2,Neuron value), (n,Neuron value) ++]
    biasVector : list # [(1,bias value), (2,bias value), (n,bias value) ++]
    activisonType :int

    dNodes : list # local derivat
    deltaNodes : list  # 
    dWeights : list # derivative of error func with respect to weights
    outs : list


    s = sigmoid()
    def __init__(self, neuronCount, weights, biases, activisonType=1):
        """initalizes the layer just sets some values

        Args:
            neuronCount (int): sets the ammount of neurons in the layer
            weights (list): a matrix that contains the weight information of the layer
            biases (list): a 'biasVector' basically a list of biases for each neuron
        """
        self.neuronCount = neuronCount
        self.weightMatrix = weights
        self.biasVector = biases
        self.activisonType = activisonType

        self.outs = []
        self.dNodes = []
        self.deltaNodes = []
        self.dWeights = []

    def calculateNeurons(self, perviousLayerNeurons : list):
        """calculate the next data vector

        Args:
            perviousLayerNeurons (list): the data vector of the perious layer

        Returns:
            [list]: the final calculated data vector for the next layer/output
        """
        neuronVector = []
        for b, w in zip(self.biasVector, self.weightMatrix): # b, w. b = biases. w = weight
            neuron = self.s.sigmoid(np.dot(w, perviousLayerNeurons) + b) # meg szorozuk a neuront a weightekkel és hozáadjuk a biast és utána sigmoid moment
            self.dNodes.append(self.s.derivative(neuron))
            neuronVector.append(neuron)
        self.outs = neuronVector
        return neuronVector


class Network:
    Layers : list
    neuronCount : list
    inputNeurons : int
    activisons : list
    
    # static acces vars
    LINEAR = 0 # activation modell id so u can acces it by Network.LINEAR/SIGMOID/TANH/RELU
    SIGMOID = 1
    TANH = 2
    RELU = 3
    
    #backpropogating variables
    
    patterns = [] #sus driping
    
    def __init__(self, inputNeurons, outputNeurons, neuronsPerLayer : list, activisons : list, debug=False):
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
        self.inputNeurons = inputNeurons
        self.Layers = []
        self.activisons = activisons
        self.neuronCount = []
        for item in neuronsPerLayer:
            self.neuronCount.append(item)
        self.neuronCount.append(outputNeurons)
        # Generation
        perv = inputNeurons
        i = 0
        for item in self.neuronCount:
            self.Layers.append(self.generateLayer(item, perv, activisons[i]))
            perv = item
            i += 1

        logging.info("Generation Complete!") # generation end

    def feed(self, inputVector : list):
        """Calculates the output of the network

        Args:
            inputVector (list): the input data of the network

        Returns:
            list: the output of the network 
        """
        dataVector = [] #filter for data oversize
        for i in range(self.inputNeurons):
            dataVector.append(inputVector[i])
        
        for item in self.Layers:
            dataVector = item.calculateNeurons(dataVector)
        return dataVector

    def exportNetwork(self, name):
        """Saves the network in a file

        Args:
            name (str): The save location for instace: /saves/save.nns
        """
        with open(name, "w+") as f:
            f.truncate(0)
            parsed = {"layers": [], }
            w = []
            b = []
            for i in range(self.inputNeurons):
                w.append([])
                b.append(0)
            parsed["layers"].append({"weights": w, "biases": b, "isInputLayer": True, "activationType": 1}) # fake input layer genration for .nns format clarity
            for item in self.Layers:
                parsed["layers"].append({"weights": item.weightMatrix, "biases": item.biasVector, "isInputLayer": False, "activationType": item.activisonType}) # general layer data
            f.write(json.dumps(parsed, indent=4, sort_keys=True))

    def importNetwork(self, name):
        """loads a saved network from file

        Args:
            name (str): the location of the save to import. for instance: /saves/save.nns
        """
        with open(name, "r") as f:
            parsed = json.load(f)
            layers = []
            self.activisons = []
            for item in parsed["layers"]:
                if item["isInputLayer"] == False:
                    layers.append(self.generateLayer(None, None, None, item)) #none is just a place holder cuz it dosent need any value but still it needs something
                    self.activisons.append(item["activationType"])
                else:
                    self.inputNeurons = len(item["weights"])
            self.Layers = layers

    def generateLayer(self, neurons, perviousNeuronCount, activisonType=1,values=None):
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
                    weights.append((random() * 2) - 1)
                weightMatrix.append(weights)
            biases = [] # bias vector generácio itt keszdödik
            for i in range(neurons):
                biases.append(0)
            return layer(neurons, weightMatrix, biases, activisonType)
        else:
            return layer(len(values["weights"]), values["weights"], values["biases"], values["activationType"])
    
    def addPattern(self, inputVector, expectedOutVector):
        self.patterns.append([inputVector, expectedOutVector])
    def clearPatterns(self):
        self.patterns = []
    def backpropogate(self, epochs : int, learningRate : int):
        for epoch in range(epochs): #epoch loop
            for layer in self.Layers: #layer meta data tisztogatás
                layer.dNodes = [] #derivált node 
                layer.deltaNodes = [] #idk 
                layer.dWeights = [] # derivált suly

            for patternIndex in range(len(self.patterns)):         #patern szelekcio
                out = self.feed(self.patterns[patternIndex][0])    # output for the cost func and the error comparison

                errors = [] #'cost' 
                dErrors = []#'derivált cost'

                for i in range(len(out)): #error calculation loop
                    errors.append((math.pow(out[i] - self.patterns[patternIndex][1][i], 2))/ 2) #fél rövidittet cost matek rész a readme ben
                    dErrors.append(out[i] - self.patterns[patternIndex][1][i]) # derivált|cost 

                for i in range(len(self.Layers))[::-1]: #back propogation loop szó szerint
                    self.Layers[i].dWeights.append([]) #minden layernek egy derivált suly és deltakimenet kell
                    self.Layers[i].deltaNodes.append([])
                    
                    if i == (len(self.Layers) -1):
                        self.Layers[i].deltaNodes[patternIndex] = np.dot(dErrors, self.Layers[i].dNodes[patternIndex])
                        #debug vars
                        dnode = self.Layers[i].dNodes[patternIndex] 
                        dnode_derr_sum = np.dot(dErrors, self.Layers[i].dNodes[patternIndex])
                        #dbve

                    elif i != 0:
                        
                        #debug variables
                        dnode2 = self.Layers[i].dNodes[patternIndex]
                        lplus1Matrix = self.Layers[i + 1].weightMatrix
                        sumlast = np.dot(self.Layers[i].dNodes[patternIndex], self.Layers[i + 1].weightMatrix)
                        deltaNodePlus1 = np.array(self.Layers[i + 1].deltaNodes[patternIndex])

                        sumarry = np.dot(self.Layers[i + 1].deltaNodes[patternIndex], np.dot(self.Layers[i].dNodes[patternIndex], self.Layers[i + 1].weightMatrix))
                        #ddve
                        self.Layers[i].deltaNodes[patternIndex] = np.dot(np.dot(self.Layers[i].dNodes[patternIndex], self.Layers[i + 1].weightMatrix), np.array(self.Layers[i + 1].deltaNodes[patternIndex]))
                    if i != 0:
                        self.Layers[i].dWeights[patternIndex] = np.outer(self.Layers[i].deltaNodes[patternIndex], self.Layers[i - 1].outs)

            for patternIndex in range(len(self.patterns)):
                for i in range(1, len(self.Layers)):
                    if self.Layers[i].dWeights[patternIndex] == []:
                        self.Layers[i].dWeights[patternIndex] = [0]

                    self.Layers[i].weightMatrix = (np.array(self.Layers[i].weightMatrix) - np.array(self.Layers[i].dWeights[patternIndex]).dot(learningRate)).tolist()
                    
                    if self.Layers[i].deltaNodes[patternIndex] == []:
                        self.Layers[i].deltaNodes[patternIndex] = [0]

                    self.Layers[i].biasVector = (np.array(self.Layers[i].biasVector) - np.array(self.Layers[i].deltaNodes[patternIndex]).dot(learningRate)).tolist()
            
        print("Backpropogation Finished!") #nice