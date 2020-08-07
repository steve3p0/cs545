# CS 545 Machine Learning
# Summer 2020 Portland State University
# Professor: Dr. Paul Doliotis
# Steve Braich
# HW #2: Neural Network

# Vectorization Libraries
import numpy as np
import pandas as pd
# TODO: USE YOUR OWN SIGMOID
from scipy.special import expit as sigmoid

# Reporting and Metrics Libraries
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sn

# Type Hinting Libraries
from nptyping import NDArray
from typing import List, Any

from tqdm import tqdm


class Network:
    """
    This class creates a neural network to recognize handwritten digits [0-9]

    TODO: update from hw1
    Example usage:
        p = Perceptron(sizes=[785, 10], train_filename=file, test_filename=file, bias=1)

        p.train(rate=0.00001, epochs=50)
        p.train(rate=0.001,   epochs=50)
        p.train(rate=0.1,     epochs=50)

    """

    # Properties Set During Compile Time
    sizes: NDArray[int]
    layers: int
    input_size: int
    hidden_size: int
    output_size: int

    # Properties set during Run Time
    η: float
    α: float
    bias: int
    epochs: int

    # Data and Labels for Training and Testing
    test_data: NDArray[Any, Any]
    test_labels: NDArray[Any]
    train_data: NDArray[Any, Any]
    train_labels: NDArray[Any]

    # Weight and Delta Matrices for the input and hidden layers
    wᵢ = NDArray[Any, Any]
    wⱼ = NDArray[Any, Any]
    Δwⱼᵢ = NDArray[Any, Any]
    Δwₖⱼ = NDArray[Any, Any]

    def __init__(self, sizes: List[int], train_filename: str=None, test_filename: str=None, bias: int=1):
        """ Constructor for Neural Network
        The constructor for this class does the following:
         (1) Initializes the layers, sizes of each layer, and the bias.
         (2) Loads in the train and test data  (this is optional - you can load it later)
         (3) Initializes the weights to an empty numpy array.
        Note: Ihis class object is set up this way so that you can instantiate a Neural Network
        object with all the necessary attributes to run multiple trainings.
        """

        # Initialize the learning rate η, the momentum α, the bias, the target, and number of epochs
        # These will be set during training
        self.η = 0.0
        self.α = 0.0
        self.epochs = 0
        self.bias = bias

        # Initialize the sizes of the layers
        self.resize(sizes)

        # Load the training samples
        # TODO: Wrap the data and labels in a set
        # TODO: Do this checking in the load function
        if train_filename is None:
            self.train_data = None
            self.train_labels = None
        else:
            self.train_data, self.train_labels = self.load(train_filename)

        # TODO: Do this checking in the load function
        if test_filename is None:
            self.test_data = None
            self.test_labels = None
        else:
            self.test_data, self.test_labels = self.load(test_filename)

    def resize(self, sizes: List[int]):
        """ Resize the Neural network
        Set the dimensions of our network

        Parameters:
            sizes   A list of layers and their sizes
        """

        self.layers = len(sizes)
        self.sizes = sizes
        self.input_size = sizes[0]
        self.hidden_size = sizes[1]
        self.output_size = sizes[len(sizes) - 1]

        # TODO: For testing purposes allow for passing weights in as a param
        # TODO: Allow this initialization to happen during training
        # The weight matrix is ultimately the output of this Perceptron
        # (specifically the train() function)
        # It is a model by which we can recognize digits
        # TODO: for consistency, input_size should be 784 pixels + 1 bias
        self.wᵢ = NDArray[self.input_size, self.hidden_size]
        # hidden size + 1 bias
        self.wⱼ = NDArray[self.hidden_size + 1, self.output_size]

        # Initialize Deltas
        # Need to Save the Deltas!!!
        # From last two pages of slide deck "Notes On Implementing NN"
        # http://web.cecs.pdx.edu/~doliotis/MachineLearningSummer2020/lectures/lecture07/NotesOnImplementingNN.pdf
        # The Delta (chainge) in weight from input layer to hidden layer, from previous iteration
        self.Δwⱼᵢ = NDArray[self.input_size, self.hidden_size]
        # The Delta (chainge) in weight from hidden layer to output layer, from previous iteration
        self.Δwₖⱼ = NDArray[self.hidden_size + 1, self.output_size]

    def load(self, filename: str) -> (NDArray[Any, Any], NDArray[Any]):
        """ Load in mnist data set
        The mnist data set file structure is as followings:

        Column   0: [0 - 9]     Represents actual digit of the image
        Cols 1-785: [0 - 255]   Represents brightness each of the 784 pixes in a 28 x 28 image

        7,0,0,0, ... , 17, 235, 250, 169, 0, 0, 0, ...
        1,0,0,0, ... , 0,    7, 251,  15, 0, 0, 0, ...
        4,0,0,0, ... , 88,   0,   0,  95, 0, 0, 0, ...
        """
        print(f"Loading Data: {filename}")

        data_file = np.loadtxt(filename, delimiter=',')
        # The bias is added to the first column
        # TODO: Try taking it out... see what happens, why do we need it?
        dataset = np.insert(data_file[:, np.arange(1, self.input_size)] / 255, 0, self.bias, axis=1)
        data_labels = data_file[:, 0]

        # TODO: I wanna wrap this into a single object
        return dataset, data_labels

    def forward(self, xᵢ: NDArray[int], hⱼ: NDArray[int]) -> (NDArray[int], NDArray[int]):
        """ Feed Forward
        For a single image vector of pixels, calculate the dot product
        to activate the next (forward) layer of neurons.

        The result will be inputs xᴷ of the next layer, hence the
        name forward.
        """
        hⱼ[1:] = sigmoid(np.dot(xᵢ, self.wᵢ))
        oₖ = sigmoid(np.dot(hⱼ, self.wⱼ))

        return hⱼ, oₖ

    def back(self, xᵢ: NDArray[float], hⱼ: NDArray[float], δⱼ: NDArray[float], δₖ: NDArray[float]):
        """ Back Propagation
        Update the weights - Need to Save the Deltas!!!
        From last two pages of slide deck "Notes On Implementing NN"
        http://web.cecs.pdx.edu/~doliotis/MachineLearningSummer2020/lectures/lecture07/NotesOnImplementingNN.pdf

            Δwⱼ,ᵢ = ηδⱼxᵢ + αΔw'ⱼ,ᵢ     where Δw'ⱼ,ᵢ is the change to this weight from previous iteration
            Δwₖ,ⱼ = ηδₖxᵢ + αΔw'ⱼ,ᵢ     where Δw'ⱼ,ᵢ is the change to this weight from previous iteration

        Parameters:
            xᵢ      The input image vector of 784 pixel values (+1 for the bias)
            hⱼ       The hidden layer vector
            δⱼ       The hidden layer error term
            δₖ       The output layer error term

        Local Variables:
            Δwⱼᵢ     The difference (delta) in weight from the input layer to the hidden layer
            Δwₖⱼ      The difference (delta) in weight from the hidden layer to the output layer
        """

        # Update input layer to hidden layer weights
        # Calculate the difference (delta) in weight from the input layer to the hidden layer
        # Use the delta from the previous iteration AND apply the momentum α
        Δwⱼᵢ = (self.η * np.outer(xᵢ, δⱼ[1:])) + (self.α * self.Δwⱼᵢ)

        # Update the weighs from the input layer to the hidden layer
        self.wᵢ += Δwⱼᵢ

        # Need to save the delta for the next iteration
        self.Δwⱼᵢ = Δwⱼᵢ

        # Update hidden layer to output layer weights
        # Calculate the difference (delta) in weight from the hidden layer to the output layer
        # Use the delta from the previous iteration AND apply the momentum α
        Δwₖⱼ = (self.η * np.outer(hⱼ, δₖ)) + (self.α * self.Δwₖⱼ)

        # Update the weighs from the hidden layer to the output layer
        self.wⱼ += Δwₖⱼ

        # Need to save the delta for the next iteration
        self.Δwₖⱼ = Δwₖⱼ

    def learn(self, hⱼ: NDArray[int], tₖ: NDArray[Any, Any]):
        """ Learn applying stochastic Gradient Decent from pages 38-40:
        http://web.cecs.pdx.edu/~doliotis/MachineLearningSummer2020/lectures/lecture04/NeuralNetworksML.pdf

        Parameters:
            hⱼ       The hidden layer
            tₖ       The target matrix tₖ for calculating output error term δₖ

        Local Variables:
            M       M is the total number of training examples
            k       k  is the indexer used in the slide deck
            xᵢ      The input  (layer) vector of 784 (+1 bias) pixels of a single image
            oₖ      The output vector
            δₖ      The output error term
            δⱼ      The hidden error term
        """

        M = len(self.train_labels)
        for k in range(0, M):
            ################################################################
            # 1. Propagate the input forward

            # Get xᵢ, the next input training example (image vector of 784 pixels + 1 bias)
            # hid_j, out_k = self.__forward(x[i, :], hid_j)
            xᵢ = self.train_data[k, :]

            # Input xᵢ to the network and compute the activation hⱼ of each hidden unit j
            # AND... Compute the activation ok of each output unit k
            hⱼ, oₖ = self.forward(xᵢ=xᵢ, hⱼ=hⱼ)

            ################################################################
            # 2. Calculate error terms

            # For each output unit k, calculate error term δₖ :
            #    δₖ ⟵ oₖ(1 - oₖ)(tₖ - oₖ)
            label = int(self.train_labels[k])
            δₖ = oₖ * (1 - oₖ) * (tₖ[label] - oₖ)

            # For each hidden unit j, calculate error term δⱼ :
            #    δⱼ ⟵ hⱼ(1 - hⱼ)(   ∑    wₖⱼ δₖ)
            #                   ᵏ ∊ ᵒᵘᵗᵖᵘᵗˢ
            δⱼ = hⱼ * (1 - hⱼ) * (np.dot(self.wⱼ, δₖ))

            ################################################################
            # 3. Update weights

            # Back propagate
            self.back(xᵢ=xᵢ, hⱼ=hⱼ, δⱼ=δⱼ, δₖ=δₖ)

    def evaluate(self, dataset: NDArray[Any, Any], data_labels: NDArray[int]) -> (float, NDArray[int]):
        """ Evaulate Accuracy
        Calculate the accuracy of the perceptron's predictions for a given dataset

        Parameters:
            dataset:        image vector of 784 pixels + 1 bias
            data_labels:    true target output
        """

        hⱼ = np.ones(self.hidden_size + 1)

        prediction = []
        for i in range(0, len(data_labels)):
            # Feed an image example forward to get a prediction vector
            _, prediction_vector = self.forward(xᵢ=dataset[i, :], hⱼ=hⱼ)

            # Add the prediction vector to the list of predictions
            prediction.append(np.argmax(prediction_vector))

        # Get an accuracy score of the prediction vs. true target labels
        accuracy = accuracy_score(data_labels, prediction)

        return accuracy, prediction

    def report(self, rate: float, prediction: List[int], train_epoch_accuracy: List[float], test_epoch_accuracy: List[float]) -> NDArray[10, 10]:
        """ Report results from training
        Display a confusion matrix and plot the accuracy

        Parameters:
            rate                    The learning rate used to train the network of perceptrons
            prediction              The values that the perceptron network predicted on the test data
            test_accuracy           The accuracy of how well the perceptron network
            train_epoch_accuracy    A list of all training accuracy scores for all epochs
            test_epoch_accuracy     A list of all testing accuracy scores for all epochs
        """

        # Confusion Matrix
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        conf_matrix = confusion_matrix(self.test_labels, prediction, labels=labels)
        df_cm = pd.DataFrame(conf_matrix, range(self.output_size), range(self.output_size))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 12})
        plt.title(f"Learning Rate {np.format_float_positional(rate, trim='-')}%")
        plt.show()

        # Graph Plot
        plt.title(f"Learning Rate {np.format_float_positional(rate, trim='-')}%")
        plt.plot(train_epoch_accuracy)
        plt.plot(test_epoch_accuracy)
        plt.ylabel("Accuracy %")
        # TODO: Make sure epoch intervals are an integer
        # https://stackoverflow.com/questions/12050393/how-to-force-the-y-axis-to-only-use-integers-in-matplotlib
        plt.xlabel("Epochs")
        plt.show()

        return conf_matrix

    def train(self, η: float=0.1, α: float=0.9, target: float=0.9, epochs: int=50, initial_weight: float=0.05) -> (NDArray[Any, Any], NDArray[Any, Any], float):
        """ Train a Neural Network to recognize handwritten digits
        Reports the accuracy and a confusion matrix
        Returns the Perceptron model in the form of weights

        Parameters:
            η                    The learning rate
            α                    The momentum
            target               Target value for tₖ
            epochs               Number of epochs
            initial_weight_low   The min range value for the initial randomized weights
            initial_weight_high  The max range value for the initial randomized weights

        Return Values:
            NDArray[785, 10]     Weights from input layer to hidden layer
            NDArray[h size, 10]  Weights from hidden layer to output layer
            float                The testing accuracy score
        """
        # Set the rate and the momentum
        self.η = η
        self.α = α

        # TODO: Make passing this in as optional (for testing)
        # Initialize weight matrices with random values
        self.wᵢ = np.random.uniform(low=(initial_weight * -1), high=initial_weight,
                                    size=(self.input_size, self.hidden_size))
        self.wⱼ = np.random.uniform(low=(initial_weight * -1), high=initial_weight,
                                    size=(self.hidden_size + 1, self.output_size))

        # Initialize weight delta matrices with zeros
        self.Δwⱼᵢ = np.zeros(self.Δwⱼᵢ.shape)
        self.Δwₖⱼ = np.zeros(self.Δwₖⱼ.shape)

        # Set the target value tₖ for output unit k to 0.9 if the input class is the kth class, 0.1 otherwise
        # specified here in Task:
        # http://web.cecs.pdx.edu/~doliotis/MachineLearningSummer2020/assignments/assignment02/Assignment02.pdf
        tₖ = np.ones((self.output_size, self.output_size), float) - target
        np.fill_diagonal(tₖ, target)

        # Initialize hidden layer
        hⱼ = np.ones(self.hidden_size + 1)

        train_epoch_accuracy = []
        test_epoch_accuracy = []

        for epoch in range(0, epochs + 1):
            # Evaluate the network's training accuracy
            train_accuracy, _ = self.evaluate(self.train_data, self.train_labels)
            train_epoch_accuracy.append(train_accuracy)
            # print(f"Epoch {epoch}:\tTraining Accuracy: {train_accuracy:.1%}")

            # Evaluate how well the network generalizes to non-training test data
            test_accuracy, _ = self.evaluate(self.test_data, self.test_labels)
            test_epoch_accuracy.append(test_accuracy)
            # print(f"\t\t\tTesting Accuracy:  {test_accuracy:.1%}")

            # Learn the weights based on the rate
            self.learn(hⱼ=hⱼ, tₖ=tₖ)

        # Evaluate Perceptron Network on Test Data
        test_accuracy, test_predictions = self.evaluate(self.test_data, self.test_labels)
        conf_matrix = self.report(rate=η, prediction=test_predictions, train_epoch_accuracy=train_epoch_accuracy, test_epoch_accuracy=test_epoch_accuracy)
        # print(f"Test Accuracy: {test_accuracy:.1%}")
        # print(f"Learning Rate: {np.format_float_positional(η, trim='-')}%")
        # print(f"Confusion Matrix: ")
        # print(conf_matrix)

        return self.wᵢ, self.wⱼ, test_accuracy
