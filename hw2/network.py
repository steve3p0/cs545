# CS 545 Machine Learning
# Winter 2020 Portland State University
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
    This class creates a network of 10 perceptrons to recognize handwritten digits [0-9]

    Example usage:
        p = Perceptron(sizes=[785, 10], train_filename=file, test_filename=file, bias=1)

        p.train(rate=0.00001, epochs=50)
        p.train(rate=0.001,   epochs=50)
        p.train(rate=0.1,     epochs=50)

    """

    # Properties Set During Compile Time
    layers: int
    input_size: int
    hidden_size: int
    output_size: int

    # Properties set during Run Time
    η: float
    α: float
    target: float
    bias: int
    epochs: int

    # Data and Labels for Training and Testing
    # TODO: Change 785 to 784?
    test_data: NDArray[Any, Any]
    test_labels: NDArray[Any]
    train_data: NDArray[Any, Any]
    train_labels: NDArray[Any]

    # Weight vectors for the input and hidden layers
    wᵢ = NDArray[Any, Any]
    wⱼ = NDArray[Any, Any]

    def __init__(self, sizes: List[int], train_filename=None, test_filename=None, bias=1):
        """ Constructor for Perceptron
        The constructor for this class does the following:
         (1) Initializes the layers, sizes of each layer, and the bias.
         (2) Loads in the train and test data  (this is optional - you can load it later)
         (3) Initializes the weights to an empty numpy array.
        Note: Ihis class object is set up this way so that you can instantiate a Neural Network
        object with all the necessary attributes to run multiple trainings.
        """

        # Set the dimensions of our network
        self.layers = len(sizes)
        self.sizes = sizes
        self.input_size = sizes[0]
        self.hidden_size = sizes[1]
        self.output_size = sizes[len(sizes) - 1]

        self.bias = bias

        # Initialize the learning rate η, the momentum α, the bias, the target, and number of epochs
        # These will be set during training
        self.η = 0.0
        self.α = 0.0
        #self.target = 0.0
        self.epochs = 0

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

        # TODO: For testing purposes allow for passing weights in as a param
        # TODO: Allow this initialization to happen during training
        # The weight matrix is ultimately the output of this Perceptron
        # (specifically the train() function)
        # It is a model by which we can recognize digits
        # TODO: for consistency, input_size should be 784 pixels + 1 bias
        self.wᵢ = NDArray[self.input_size, self.hidden_size]
        # hidden size + 1 bias
        self.wⱼ = NDArray[self.hidden_size + 1, self.output_size]

    def load(self, filename: str) -> (NDArray[Any, 785], NDArray[Any]):
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

    # def forward(self, xᴷ: NDArray[785]) -> NDArray[10]:
    #                           input            hidden           hidden         output
    def forward(self, xᵢ: NDArray[int], hⱼ: NDArray[int]) -> (NDArray[int], NDArray[int]):
        """ Feed Forward
        For a single image vector of pixels, calculate the dot product
        to activate the next (forward) layer of neurons.

        The result will be inputs xᴷ of the next layer, hence the
        name forward.
        """

        # Remember the bias in included in the first column of every row: self.weights[x][0]
        #activation_vector = np.dot(xᴷ, self.weights)
        #return activation_vector

        # [1:] means index 1 to the end (don't change index 0 - bias)
        # hⱼ[1:] = sigmoid(xᵢ.dot(self.wᵢ))
        hⱼ[1:] = sigmoid(np.dot(xᵢ, self.wᵢ))

        #oₖ = sigmoid(hⱼ.dot(self.wⱼ))
        oₖ = sigmoid(np.dot(hⱼ, self.wⱼ))

        return hⱼ, oₖ

    def _old_back(self, k: int, xᴷ: NDArray[785], yᴷ: NDArray[10], η: float) -> NDArray[785, 10]:
        """ Back Propagation
        Update the weights by minimizing the cost that the weights will produce with the next sample
        This is according to the Perceptron Slide deck on page 34-35:
        http://web.cecs.pdx.edu/~doliotis/MachineLearningSummer2020/lectures/lecture02/PerceptronsML.pdf

             wᵢ ⟵ wᵢ + Δwᵢ
            Δwᵢ = η(tᴷ - yᴷ)xᵢᴷ

        Parameters:
            k       The indexer used in perceptron learning algorithm
            xᴷ      The input image vector of 784 pixel values (+1 for the bias)
            yᴷ      The output vector (result of w * x)
            η       The Learning rate

        Local Variables:
            t       The next target label (true value of output)
            tᴷ      Vectorized output layer of target value
            yᴷ      Modify the output vector using ArgMax to zero out outputs except highest value
            Δwᵢ     The gradient or vector derivative in order to minimize the cost function
        """

        # Get the target label: the true value [0-9] of the training sample image)
        t = int(self.train_labels[k])

        # Create a target vector of ten elements, light up the correct value t
        tᴷ = np.insert(np.zeros((1, self.output_size - 1)), t, 1)

        # Create a vector for the predicted value
        yᴷ = np.insert(np.zeros((1, self.output_size - 1)), np.argmax(yᴷ), 1)

        # Find the gradient of the weights
        # Δwᵢ= η(tᴷ - yᴷ)xᵢᴷ

        # Δwᵢ = η * np.dot(tᴷ - yᴷ, xᴷ)
        # Δwᵢ = η * np.dot(xᴷ, (tᴷ - yᴷ))
        # Δwᵢ = η * np.dot(np.reshape(xᴷ, (self.input_size, 1)), np.reshape(tᴷ - yᴷ, (1, self.output_size)))
        # Δwᵢ = η * np.dot(np.reshape(tᴷ - yᴷ, (1, self.output_size)), np.reshape(xᴷ, (self.input_size, 1)))
        Δwᵢ = η * np.dot(np.reshape(xᴷ, (self.input_size, 1)), np.reshape(tᴷ - yᴷ, (1, self.output_size)))

        # what is the problem here?
        #   xᴷ        ndarray: (785,) [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0., 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0., 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0., 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0., 0. 0. 0. 0.]
        #   tᴷ - yᴷ   ndarray: (10, ) [-0.0019597   0.08868587  0.9632121  -0.01701219 -0.06978089 -0.09443413, -0.2448718  -0.08801273 -0.19624884  0.13512257]

        # So when we fuck with this:
        #   xᴷ        becomes   np.reshape(xᴷ, (self.input_size, 1))
        #             {ndarray: (785, 1)} [[1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.
        #
        #   tᴷ - yᴷ  becomes   np.reshape(tᴷ - yᴷ, (1, self.output_size))
        #            {ndarray: (1, 10)}   [[-0.0019597   0.08868587  0.9632121  -0.01701219 -0.06978089 -0.09443413,  -0.2448718  -0.08801273 -0.19624884  0.13512257]]

        # blash
        #
        # [[-1.95969690e-08  8.86858666e-07  9.63212100e-06 -1.70121873e-07,  -6.97808933e-07 -9.44341300e-07 -2.44871796e-06 -8.80127347e-07,  -1.96248841e-06  1.35122575e-06], [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00,   0.00000000e+00  0.00


        self.weights += Δwᵢ

        return self.weights


    # self.weights = self.back(k, xᴷ, yᴷ, η)
    #
    # Back propagate
    # xᵢ[k, :]   : x[i, :] (train pixel inputs)
    # hⱼ : hid_j
    # δₖ : error_o
    # δⱼ : error_h
    def back(self, xᵢ: NDArray[int], hⱼ: NDArray[int], δⱼ: NDArray[int], δₖ: NDArray[int]):

        # Comupte delta in first layer
        Δwⱼᵢ = (self.η * np.outer(xᵢ, δⱼ[1:])) + (self.α * self.wᵢ)

        # Update weights in first layer
        self.wᵢ += Δwⱼᵢ

        # Compute delta in second layer
        Δwₖⱼ = (self.η * np.outer(hⱼ, δₖ)) + (self.α * self.wⱼ)

        # Update weights in second layer
        self.wⱼ += Δwₖⱼ

    def learn(self, η: float, hⱼ: NDArray[int], tₖ: NDArray[Any, Any]):
        """ The Perceptron Learning Algorithm
        Iterate thru all training examples, feeding the outputs forward and back propagating the
        updated weights. This function tries to exactly model the algorithm as it is described
        in this slide on page 35:

        http://web.cecs.pdx.edu/~doliotis/MachineLearningSummer2020/lectures/lecture02/PerceptronsML.pdf

             wᵢ ⟵ wᵢ + Δwᵢ
            Δwᵢ  = η(tᴷ - yᴷ)xᵢᴷ

        Parameters:
            η       The Learning rate

        Local Variables:
            M       M is the total number of training examples
            k       k  is the indexer used in the slide deck
            xᴷ      xᴷ is the input  (layer) vector of 784 (+1 bias) pixels of a single image
            yᴷ      yᴷ is the output (layer) vector of 10 nodes representing activation level of a digit
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
            # xᵢ[k, :]   : x[i, :] (train pixel inputs)
            # hⱼ : hid_j
            # δₖ : error_o
            # δⱼ : error_h

            self.back(xᵢ=xᵢ, hⱼ=hⱼ, δⱼ=δⱼ, δₖ=δₖ)

        #return self.wᵢ, self.wⱼ

    def evaluate(self, dataset: NDArray[785], data_labels: NDArray[int]) -> (float, NDArray[int]):
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

            # def forward(self, xᵢ: NDArray[int], hⱼ: NDArray[int]) -> (NDArray[int], NDArray[int]):
            # hⱼ, oₖ = self.forward(xᵢ=xᵢ, hⱼ=hⱼ)
            _, prediction_vector = self.forward(xᵢ=dataset[i, :], hⱼ=hⱼ)

            # Add the prediction vector to the list of predictions
            prediction.append(np.argmax(prediction_vector))

        # Get an accuracy score of the prediction vs. true target labels
        accuracy = accuracy_score(data_labels, prediction)

        return accuracy, prediction

    def report(self, rate: float, prediction: List[int], test_accuracy: float,
                     train_epoch_accuracy: List[float], test_epoch_accuracy: List[float]) -> NDArray[10, 10]:
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

    # TODO: Get rid of hardcoded hint values -> (NDArray[785, 10]
    def train(self, η: float, α:float, target: float, epochs=50, initial_weight_low=-.05, initial_weight_high=.05) -> (NDArray[Any, Any], NDArray[Any, Any], float):
        """ Perceptron Network Training
        Train 10 perceptrons to recognize handwritten digits
        Reports the accuracy and a confusion matrix
        Returns the Perceptron model in the form of weights in a 785 x 10 matrix

        Parameters:
            rate                    The learning rate
            epochs                  The total number of training interations to run over all the training samples
            initial_weight_low      The min range value for the initial randomized weights
            initial_weight_high     The max range value for the initial randomized weights

        Return Values:
            NDArray[785, 10]        Perceptron model in the form of weights in a 785 x 10 matrix
            float                   The testing accuracy score
        """
        # Set the rate and the momentum
        self.η = η
        self.α = α

        # TODO: Make passing this in as optional (for testing)
        # Initialize weight matrices with random values
        self.wᵢ = np.random.uniform(low=initial_weight_low, high=initial_weight_high, size=(self.input_size, self.hidden_size))
        self.wⱼ = np.random.uniform(low=initial_weight_low, high=initial_weight_high, size=(self.hidden_size, self.output_size))

        # Set the target value t k for output unit k to 0.9 if the input class is the kth class, 0.1 otherwise
        tₖ = np.ones((self.output_size, self.output_size), float) - target

        # Initialize hidden layer
        hⱼ = np.ones(self.hidden_size + 1)

        train_epoch_accuracy = []
        test_epoch_accuracy = []

        for epoch in range(0, epochs + 1):
            # Evaluate the network's training accuracy
            train_accuracy, _ = self.evaluate(self.train_data, self.train_labels)
            train_epoch_accuracy.append(train_accuracy)
            print(f"Epoch {epoch}:\tTraining Accuracy: {train_accuracy:.1%}")

            # Evaluate how well the network generalizes to non-training test data
            test_accuracy, _ = self.evaluate(self.test_data, self.test_labels)
            test_epoch_accuracy.append(test_accuracy)
            print(f"\t\t\tTesting Accuracy:  {test_accuracy:.1%}")

            # Learn the weights based on the rate
            # learn(self, η: float, hⱼ: NDArray[int], tₖ: NDArray[int, int])
            self.learn(η=η, hⱼ=hⱼ, tₖ=tₖ)
            #self.learn(η, hⱼ)

        # Evaluate Perceptron Network on Test Data
        test_accuracy, test_predictions = self.evaluate(self.test_data, self.test_labels)

        # Report Accuracy and Confusion Matrix
        #     def report(self, rate: float, prediction: List[int], test_accuracy: float,
        #                      train_epoch_accuracy: List[float], test_epoch_accuracy: List[float]) -> NDArray[10, 10]:
        conf_matrix = self.report(rate=η, test_predictions=test_predictions, test_accuracy=test_accuracy,
                                          train_epoch_accuracy=train_epoch_accuracy, test_epoch_accuracy=test_epoch_accuracy)
        print(f"\n")
        print(f"Test Accuracy: {test_accuracy:.1%}")
        print(f"Learning Rate: {np.format_float_positional(η, trim='-')}%")
        print(f"Confusion Matrix: ")
        print(conf_matrix)

        # return self.weights, test_accuracy
        return self.wᵢ, self.wⱼ, test_accuracy
