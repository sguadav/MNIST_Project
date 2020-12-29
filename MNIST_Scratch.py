import numpy as np
import scipy

class Neural_Network:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        """This is the intialization function for the Neural Network Network

        Args:
            inputNodes (int): Input nodes for the input layer of the NN layer
            hiddenNodes (int): Number of hidden neurons in the hidden layer
            outputNodes (int): Output neurons in the output layers (0-9 in this case of)
            learningRate (float): Explanatory
        """
        self.input_nodes = inputNodes
        self.hidden_nodes = hiddenNodes
        self.output_nodes = outputNodes

        print("input: ", self.input_nodes, ", hidden: ", self.hidden_nodes, ", output: ", self.output_nodes)

        # Linking the weight matrices: wih and who
        #self.wih = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        #self.who = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

        # Try later
        self.wih = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        print("Matrix 1: \n", self.wih)
        print("Matrix 2: \n", self.who)

        #learning rate
        self.learning_rate= learningRate

        # Activation function, using the sigmoid function (Values of each neuron)
        self.activation_function = lambda x: scipy.special.expit(x)

    
    def train():
        pass

    def query():
        pass

example = Neural_Network(3,5,2,0.2)