import numpy as np
import scipy

class Neural_Network:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        """This is the intialization function for the Neural Network Networks

        Args:
            inputNodes (int): Input nodes for the input layer of the NN layer (All the pixels)
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
        # wih: weigth input to hidden layers
        # who: weight hidden to output layer
        self.wih = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        print("Matrix 1: \n", self.wih)
        print("Matrix 2: \n", self.who)

        #learning rate
        self.learning_rate= learningRate

        # Activation function, using the sigmoid function (Values of each neuron)
        self.activation_function = lambda x: scipy.special.expit(x)

    
    def train(self, input_list, target_list):
        """In this function is where we train our Neural Network to deal with the data set, doing both forward and backward
        propagation


        Args:
            input_list (list): list of inputs (numbers to evaluate)
            target_list (list): list of expected results
        """
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T 

        # Calculating and analyzing the Forward propagation
        # Calculate signals into the hidden layers
        hidden_inputs = np.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        #output layer error is the target - actual
        output_errors = targets - final_outputs
        # hidden layer errors is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Back propagation
        # Update the weights for the links between the hidden and output layers
        self.who += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


    def use(self, input_list):
        """Using the NN after training it

        Args:
            input_list (input): Complete list of all the input(in this case is all the pixels in the image)
        """
        inputs = np.array(input_list, ndmin=2).T 

        # Calculate signals into hidden layers
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate the signals emerging from hidden layers
        hidden_outputs = self.activation_function(hidden_inputs)

        #Calculate signals into final output
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        

example = Neural_Network(3,5,2,0.2)