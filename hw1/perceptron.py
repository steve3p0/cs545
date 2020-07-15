# CS 545 Machine Learning
# Professor: Dr. Paul Doliotis
# HW #1: Perceptron
from typing import List, Any

import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from nptyping import NDArray
from typing import Any


class Perceptron:

    weights: List[Any]

    def __init__(self, sizes: List[int], train_filename, test_filename, epochs=50, bias=1):
        """ Constructor for Perceptron
        The constructor for this class does the following:
         (1) Iinitializes the layers, sizes of each layer, the number of epochs, and the bias.
         (2) Loads in the train and test data
         (3) Initializes the weights to an empty list.
        Note: Ihis class object is set up this way so that you can instantiate a Perceptron
        object with all the necessary attributes to run multiple trainings.

        When you run the train() method, every class property is persisted except for weights

            p = Perceptron(sizes=[785, 10], train_filename=file, test_filename=file, epochs=50, bias=1)
            p.train(rate=0.00001)
            p.train(rate=0.001)
            p.train(rate=0.1)

        You can also override optional parameters epoch and bias when you train as well:

            p.train(rate=0.1)
            p.train(rate=0.1, epoch=100, bias=.1)

        """
        self.layers = len(sizes)
        self.sizes = sizes
        self.input_size = sizes[0]
        self.output_size = sizes[len(sizes) - 1]
        self.bias = bias

        # TODO: This should be initialized to 0 here
        # The number of epochs and the learning rate should be something
        # we set during training
        self.epochs = epochs
        self.rate = 0.0

        # TODO: this is kind of useless right now.
        # I kind of want to separate the bias inputs from the test inputs
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # TODO: Wrap the data and labels in a set
        self.train_data, self.train_labels = self.load(train_filename)
        self.test_data, self.test_labels = self.load(test_filename)

        # The weight matrix is ultimately the output of this Perceptron
        # (specifically the train() function)
        # It is a model by which we can recognize digits
        self.weights = NDArray[785, 10]

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

    def forward(self,  xᴷ: NDArray[785]) ->  NDArray[10]:
        """ Feed Forward
        For a single image vector of pixels, calculate the dot product
        to activate the next (forward) layer of neurons.

        The result will be inputs (x_k) of the next layer, hence the
        name forward.
        """

        # Remember the bias in included in the first column of every row: self.weights[x][0]
        activation_vector = np.dot(xᴷ, self.weights)

        return activation_vector

    def back(self, k: int, xᴷ: NDArray[785], yᴷ: NDArray[10], η: float) -> NDArray[785, 10]:
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

    def learn(self, η: float) -> NDArray[785, 10]:
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

            # Get xᴷ, the next input training example (image vector of 784 pixels + 1 bias)
            xᴷ = self.train_data[k]

            # Get yᴷ, the output vector by applying the weights to xᴷ and feeding that forward to the next layer
            yᴷ = self.forward(xᴷ)

            # Update the weights using the Perceptron training Algorithm and propagate them back
            self.weights = self.back(k, xᴷ, yᴷ, η)

        return self.weights

    def evaluate(self, dataset, data_labels) -> (float, np.ndarray):
        """ Evaulate Accuracy
        Calculate the accuracy of the perceptron's predictions for a given dataset
            dataset:        image vector of 784 pixels + 1 bias
            data_labels:    tru target output
        """

        prediction = []
        for i in range(0, len(data_labels)):
            # Feed an image example forward to get a prediction vector
            prediction_vector = self.forward(dataset[i, :])

            # Add the prediction vector to the list of predictions
            prediction.append(np.argmax(prediction_vector))

        # Get an accuracy score of the prediction vs. true target labels
        accuracy = accuracy_score(data_labels, prediction)

        return accuracy, prediction

    def report(self, test_accuracy, learn_rate, prediction, arr_train_acc, arr_test_acc):
        # Confusion Matrix
        print("\t\tTest Set Accuracy = " + str(test_accuracy) +
              "\n\nLearning Rate = " + str(learn_rate) +
              "\n\nConfusion Matrix :\n")
        conf_matrix = confusion_matrix(self.test_labels, prediction)
        print(confusion_matrix(self.test_labels, prediction))
        print("\n")

        df_cm = pd.DataFrame(conf_matrix, range(self.output_size), range(self.output_size))
        sn.set(font_scale=1.4)
        # font size
        sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 12})

        #plt.title("Learning Rate %r" % learn_rate)
        plt.title(f"Learning Rate {learn_rate}")
        plt.show()

        # Graph Plot
        #plt.title("Learning Rate %r" % learn_rate)
        plt.title(f"Learning Rate {learn_rate}")
        plt.plot(arr_train_acc)
        plt.plot(arr_test_acc)
        plt.ylabel("Accuracy %")
        plt.xlabel("Epochs")
        plt.show()

    def train(self, rate: float, epoch=1, bias=1) -> NDArray[785, 10]:
        """
        Train 10 perceptrons to recognize handwritten digits
        Reports the accuracy and a confusion matrix
        Returns the Perceptrons in the form of a 785 x 10 matrix

        rate:    The learning rate
        epoch:   An iteration of training over all training examples
        bias:    A way to get the perceptron to fire
        """

        # Initialize weight matrix with random values
        self.weights = np.random.uniform(low=-0.05, high=0.05, size=(self.input_size, self.output_size))

        epoch = 0
        arr_epoch = []
        arr_test_acc = []
        arr_train_acc = []

        while (1):
            train_accuracy, pred = self.evaluate(self.train_data, self.train_labels)
            print(f"Epoch {str(epoch)} :\tTraining Set Accuracy = {str(train_accuracy)}")
            if epoch == self.epochs:
                # If network is converged, stop training
                break

            # Evaluate the usefulness of the perceptrons
            test_accuracy, pred = self.evaluate(self.test_data, self.test_labels)
            print(f"\t\tTest Set Accuracy = {str(test_accuracy)}")

            # What are we doing with this?
            prev_accu = train_accuracy

            epoch += 1

            # Train the network
            self.weights = self.learn(rate)

            arr_epoch.append(epoch)
            arr_train_acc.append(train_accuracy)
            arr_test_acc.append(test_accuracy)

        # Test network on test set and get test accuracy
        test_accu, pred = self.evaluate(self.test_data, self.test_labels)

        # Report Accuracy and Confusion Matrix
        self.report(test_accu, rate, pred, arr_train_acc, arr_test_acc)

        return self.weights

train_file = 'mnist_train.csv'
test_file = 'mnist_validation.csv'
p = Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, epochs=3, bias=1)
p.train(rate=0.00001)
p.train(rate=0.001)
p.train(rate=0.1)
