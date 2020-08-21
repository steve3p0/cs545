# CS 545 Machine Learning
# Summer 2020 Portland State University
# Professor: Dr. Paul Doliotis
# Steve Braich
# HW #4: K-mean Clustering

import sys
import numpy as np
from random import randint
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from PIL import Image

# Type Hinting Libraries
from nptyping import NDArray
from typing import List, Any


class Kmeans:
    """ K-means Cluster Class Object
    """

    k: int
    train_data: []
    test_data: []
    centroids: []
    labels: []

    mean_square_error: float
    mean_square_separation: float
    mean_entropy: float
    accuracy: float
    predictions: float

    def __init__(self, k, trainfile: str = None, testfile: str = None):
        """ Constructor for Kmeans object """
        self.k = k
        self.train_data = self.load(trainfile)
        self.test_data = self.load(testfile)

        self.centroids = []
        self.labels = []

        self.mean_square_error = 0.0
        self.mean_square_separation = 0.0
        self.mean_entropy = 0.0
        self.accuracy = 0.0
        self.predictions = 0.0

    @staticmethod
    def load(datafile: str) -> []:
        """ Load the data from a file """

        data = np.loadtxt(datafile, delimiter=',')

        return data

    ####################################################################################
    # TRAINING FUNCTIONS #############################################################
    # Find the best cluster out of 5 randomly generated centroids
    def train(self) -> [float]:
        """ Train K-means Clustering Model
        Returns a set of centroids
        """
        mse_prev = 0
        assignments = []
        self.centroids = np.zeros(self.k)

        # TODO: this range shouldn't be fixed?
        # DO UNTIL?
        for i in range(5):
            cds = self.create_centroids(self.train_data[:, :-1])

            # BIG FUCKING NO-NO
            for i in range(100):
                cds, assignments = self.train_centroids(cds, self.train_data)

            mss = self.find_sss(cds)
            mse = self.find_sse(cds, self.train_data, assignments)

            # if we have a reduction in error, safe the model
            if self.mean_square_error < mse_prev or mse_prev == 0:
                self.centroids = cds
                self.mean_square_separation = mss
                self.mean_square_error = mse

            mse_prev = mse

        self.mean_entropy = self.find_entropy(self.train_data, assignments)

        self.labels, self.predictions = self.predict(self.train_data)
        self.accuracy = metrics.accuracy_score(self.train_data[:, -1], self.predictions)

        return self.centroids

    def retrain_centroids(self, centroids, assignments, data) -> [float]:
        """ Retrain Centroids
        TODO: Rewrite - Find new centroids based on their assignments and data
        """

        for i in range(len(centroids)):
            centroids[i] = np.mean(data[assignments == i, :-1], axis=0)

        return centroids

    def create_centroids(self, train_data: []) -> [float]:
        """ Create Centroids
        Basic creation of centroids from the data
        """

        centroids = np.zeros([self.k, 64], dtype=float)

        for i in range(len(centroids)):
            centroids[i] = train_data[(randint(0, len(train_data))), :]

        return centroids

    def train_centroids(self, centroids: NDArray[float], data: NDArray[float]) -> ([], []):
        """ Train Centroids
        The initial "train" or making new centroids based on the data
        """

        distances = self.all_euclidean_dist(centroids, data)
        assignments = self.assign_clusters(distances)
        centroids = self.retrain_centroids(centroids, assignments, data)

        return centroids, assignments

    ####################################################################################
    # TODO: RENAME METHOD
    def predict(self, data: NDArray[float]) -> ([], []):
        """ Predict the centroids
        REWORD
        Find the predictions for the centroids and the data almost the same as pred_test_results
        """
        distance = self.all_euclidean_dist(self.centroids, data)
        assignments = self.assign_clusters(distance)
        centroid_labels = np.zeros(self.k)
        predictions = np.zeros(data.shape[0])

        for i in range(len(self.centroids)):
            centroid_labels[i] = stats.mode(data[assignments == i, -1])[0]
            predictions[assignments == i] = centroid_labels[i]

        return centroid_labels, predictions

    def pred_test_results(self, centroids: NDArray[float], data: NDArray[float], labels: []) -> []:
        """ Predict where what cluster the test data is in """

        distance = self.all_euclidean_dist(centroids, data)
        assignments = self.assign_clusters(distance)
        test_predictions = np.zeros(data.shape[0])

        for i in range(len(centroids)):
            test_predictions[assignments == i] = labels[i]

        return test_predictions

    ####################################################################################
    # MATH SHIT ######################################################################

    @staticmethod
    def euclidean_dist(centroid: float, data_point: float) -> float:
        """ Euclidean Distance
        Calculate the Eculidean distance of a data point and a centroid
        """

        row = np.sqrt(np.sum((centroid - data_point) ** 2, axis=1))

        return row

    #def all_euclidean_dist(self, centroids: [], data: []) -> NDArray[Any, Any]:
    def all_euclidean_dist(self, centroids: [], data: []) -> NDArray[float]:
        """ Euclidean distances of all data """
        all_e_dist = np.zeros([len(data), len(centroids)])

        for i in range(len(centroids)):
            all_e_dist[:, i] = self.euclidean_dist(centroids[i, :], data[:, :-1])

        return all_e_dist

    def find_sse(self, centroids: [], data: [], assign) -> float:
        """ Sum of Squares Error
        Calculate the Sum of Squares Error of all the centroids """

        sse = 0

        for i in range(self.k):
            sse += np.sum(self.euclidean_dist(centroids[i], data[assign == i, :-1]) ** 2)

        return sse

    def find_sss(self, centroids: []) -> float:
        """ Mean Sum of Separation
        Find the SSS of the centroids.
        """

        sss = 0.0

        for i in range(self.k):
            for j in range(self.k):
                sss = sss + (centroids[i] - centroids[j]) ** 2

        sss = np.sum(sss)

        return sss

    def find_entropy(self, data: [], assignment: [int]) -> float:
        """ Find the Mean Entropy
        Find the Entropy of each cluster
        """

        ratios = np.zeros(10)
        entropy = np.zeros(self.k)
        m_entropy = 0

        for i in range(self.k):
            for j in range(10):
                ratios[j] = float(len(data[(assignment == i) & (data[:, -1] == j)])) / float(len(data[assignment == i]))

            log_ratio = np.log2(ratios)
            log_ratio[log_ratio == np.log2(0)] = 0
            entropy[i] = -np.sum(ratios * log_ratio)
            m_entropy += entropy[i] * len(data[assignment == i])

        m_entropy /= len(data)

        return m_entropy

    ####################################################################################

    @staticmethod
    def assign_clusters(distances) -> []:
        """ Assign a label to a cluster """

        assignments = np.zeros([len(distances)])

        for i in range(len(distances)):
            assignments[i] = np.argmin(distances[i])

        return assignments

    def evaluate(self, data: []) -> (float, []):
        """ Evalute Clustering Model
        """

        # Run the testing data against the clusters and gain predictions
        predictions = self.pred_test_results(self.centroids, data, self.labels)

        # The accuracy for the testing data
        acc = metrics.accuracy_score(data[:, -1], predictions)

        # Confusion matrix!
        confusion_matrix = metrics.confusion_matrix(data[:, -1], predictions)

        return acc, confusion_matrix

    def report(self, test_accuracy: float, confusion_matrix: [[]]) -> None:
        """ Report Evaluations Results
        Report results from evaluation of test data and metrics on training
        """
        print("These are the corresponding labels for each cluster:", self.labels)
        print("This is the accuracy for the training data: ", self.accuracy * 100)
        print("This is the accuracy for the testing data: ", test_accuracy * 100)

        print("This is the best SSS: ", self.mean_square_separation)
        print("This is the best SSE: ", self.mean_square_error)
        print("The mean Entropy is: ", self.mean_entropy)

        print("The confusion matrix: ")
        print(confusion_matrix)

        if self.k == 10:
            for i in range(self.k):
                plt.imsave('exp1_image_%i.png' % i, np.array(self.centroids[i, :]).reshape(8, 8), cmap=cm.gray)
        if self.k == 30:
            for i in range(self.k):
                plt.imsave('exp2_image_%i.png' % i, np.array(self.centroids[i, :]).reshape(8, 8), cmap=cm.gray)


if __name__ == "__main__":

    # Check Command line args
    if len(sys.argv) != 4:
        print("Incorrect number of command line arguments.\n")
        print("Usage: python kmeans.py K train_file_path test_file_path")
        sys.exit(0)

    km = Kmeans(k=sys.argv[1], trainfile=sys.argv[2], testfile=sys.argv[3])
    km.train()
    accuracy, conf_matrix = km.evaluate(data=km.test_data)
    km.report(test_accuracy=accuracy, confusion_matrix=conf_matrix)
