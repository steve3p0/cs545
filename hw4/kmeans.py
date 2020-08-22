# CS 545 Machine Learning
# Summer 2020 Portland State University
# Professor: Dr. Paul Doliotis
# Steve Braich
# HW #4: K-mean Clustering

import sys
import numpy as np
import pandas as pd
from random import randint
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

# Type Hinting Libraries
from nptyping import NDArray
from typing import List, Any


class Kmeans:
    """ K-means Cluster Class Object
    """

    k: int
    train_data: NDArray[float]
    test_data: NDArray[float]
    centroids: NDArray[float]
    labels: NDArray[int]

    mean_square_error: float
    mean_square_separation: float
    mean_entropy: float
    accuracy: float
    predictions: NDArray[float]

    def __init__(self, k, trainfile: str = None, testfile: str = None):
        """ Constructor for Kmeans object """
        self.k = k
        self.train_data = self.__load(trainfile)
        self.test_data = self.__load(testfile)

        self.centroids = np.zeros(self.k)
        self.labels = np.zeros(self.k)

        self.mean_square_error = 0.0
        self.mean_square_separation = 0.0
        self.mean_entropy = 0.0
        self.accuracy = 0.0
        # self.predictions = NDArray[float]

    @staticmethod
    def __load(datafile: str) -> NDArray[Any, Any]:
        """ Load the data from a file """
        data = np.loadtxt(datafile, delimiter=',')
        return data

    # TODO: SHOULD THIS REALLY BE A STATIC METHOD?
    # AGAIN - THIS SHOULD BE A ONE-LINER
    @staticmethod
    def label_clusters(distances: NDArray[float]) -> NDArray[float]:
        """ Assign a label to a cluster """
        assignments = np.zeros([len(distances)])

        for i in range(len(distances)):
            assignments[i] = np.argmin(distances[i])

        return assignments

    def train(self) -> NDArray[float]:
        """ Train K-means Clustering Model
        Choose the run (out of 5) that yields the smallest average mean-square-error (mse).
        Repeat the following 5 times, with different random number seeds to compute the
        original (random) cluster centers
        """

        mse_prev = 0
        clusters = []

        for i in range(5):
            # Get everything in the training data, except the last column
            tmp_centroids = np.zeros([self.k, 64], dtype=float)
            for j in range(len(tmp_centroids)):
                tmp_centroids[j] = self.train_data[(randint(0, len(self.train_data))), :-1]

            # Stop iterating K-Means when all cluster centers stop changing.
            prev_clusters = []
            while True:
                distances = self.all_euclidean_dist(tmp_centroids, self.train_data)
                clusters = self.label_clusters(distances)

                # retrain centroids
                for k in range(len(tmp_centroids)):
                    tmp_centroids[k] = np.mean(self.train_data[clusters == k, :-1], axis=0)

                # if there is no change in the cluster, stop!
                if np.array_equal(prev_clusters, clusters):
                    break
                prev_clusters = clusters

            mss = self.find_sss(tmp_centroids)
            mse = self.average_mean_square_error(tmp_centroids, self.train_data, clusters)

            # Choose the run (out of 5) that yields the smallest average mean-square-error (mse)
            if self.mean_square_error < mse_prev or mse_prev == 0:
                self.centroids = tmp_centroids
                self.mean_square_separation = mss
                self.mean_square_error = mse

            mse_prev = mse

        # Save metrics for training
        self.mean_entropy = self.find_entropy(self.train_data, clusters)
        self.labels, self.predictions = self.predict(self.centroids, self.train_data, self.labels)
        self.accuracy = metrics.accuracy_score(self.train_data[:, -1], self.predictions)

        # The centriods are "the model" for k-means
        return self.centroids

    def predict(self, centroids: NDArray[float], data: NDArray[float], labels: NDArray[float]) -> (NDArray[float], NDArray[float]):
        """ Predict the centroids
        REWORD
        Find the predictions for the centroids and the data
        """
        distance = self.all_euclidean_dist(centroids, data)
        assignments = self.label_clusters(distance)
        predictions = np.zeros(data.shape[0])

        for i in range(len(centroids)):
            labels[i] = stats.mode(data[assignments == i, -1])[0]
            predictions[assignments == i] = labels[i]

        return labels, predictions

    def evaluate(self, data: NDArray[float]) -> (float, NDArray[float]):
        """ Evalute Clustering Model
        """

        # Run the testing data against the clusters and gain predictions
        _, predictions = self.predict(self.centroids, data, self.labels)

        # The accuracy for the testing data
        acc = metrics.accuracy_score(data[:, -1], predictions)

        # Confusion matrix!
        confusion_matrix = metrics.confusion_matrix(data[:, -1], predictions)

        return acc, confusion_matrix

    def report(self, test_accuracy: float, confusion_matrix: [[]]) -> None:
        """ Report Evaluations Results
        Report results from evaluation of test data and metrics on training
        """
        print(f"\n\nLabeled Clusters:")
        print(f"\t{self.labels.astype(int)}")
        print(f"Training Accuracy: {self.accuracy * 100:.2f}%")
        print(f"Testing  Accuracy: {test_accuracy * 100:.2f}%")

        print("\nTraining Metrics:")
        print(f"\tAverage Mean Square Error: {self.mean_square_error}")
        print(f"\tMean Square Separation:    {self.mean_square_separation}")
        print(f"\tMean Entropy:              {self.mean_entropy}")

        print("\nTesting Metrics:")
        print("\nConfusion Matrix: ")
        print(confusion_matrix)
        # Confusion Matrix
        #df_cm = pd.DataFrame(confusion_matrix, range(len(self.labels)), range(len(self.labels)))
        df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 12})
        plt.title(f"K = {self.k}")
        plt.show()

        # Show the Image
        for i in range(self.k):
            data = np.array(self.centroids[i, :]).reshape(8, 8)
            extent = (0, data.shape[1], data.shape[0], 0)
            plt.imshow(X=data, cmap='gray', interpolation='none', extent=extent)
            plt.title(f"K = {self.k},  Label = {int(self.labels[i])}")
            ax = plt.gca()

            # Major ticks
            # ax.set_xticks(np.arange(0, 7, 1));
            # ax.set_yticks(np.arange(0, 7, 1));
            ax.set_xticks(np.arange(1, 8, 1))
            ax.set_yticks(np.arange(1, 8, 1))

            # Labels for major ticks
            ax.set_xticklabels(np.arange(1, 8, 1))
            ax.set_yticklabels(np.arange(1, 8, 1))

            plt.show()

    ####################################################################################
    # MATH SHIT ######################################################################

    @staticmethod
    def euclidean_dist(centroids: NDArray[float], data: NDArray[float]) -> NDArray[float]:
        """ Euclidean Distance
        Calculate the Eculidean distance of a data point and a centroid
        """

        row = np.sqrt(np.sum((centroids - data) ** 2, axis=1))
        # row = np.sqrt((centroids - data) ** 2)
        # row = np.sum((centroids - data) ** 2, axis=1)
        # row = np.sqrt(np.sum((centroids - data), axis=1))

        return row

    #def all_euclidean_dist(self, centroids: [], data: []) -> NDArray[Any, Any]:
    def all_euclidean_dist(self, centroids: NDArray[float], data: NDArray[float]) -> NDArray[float]:
        """ Euclidean distances of all data """

        # #dist = numpy.linalg.norm(a - b)
        # a = centroids
        # b = data[:, :-1]
        # all_e_dist = np.linalg.norm(a - b)

        all_e_dist = np.zeros([len(data), len(centroids)])

        for i in range(len(centroids)):
            all_e_dist[:, i] = self.euclidean_dist(centroids[i, :], data[:, :-1])

        return all_e_dist

        # all_e_dist = np.zeros([len(data), len(centroids)])
        #
        # for i in range(len(centroids)):
        #     all_e_dist[:, i] = np.linalg.norm(centroids[i, :] - data[:, :-1])
        #
        # return all_e_dist

    def average_mean_square_error(self, centroids: NDArray[float], data: NDArray[float], assign) -> float:
        """ Average Mean Square Error
        Calculate the Average Mean Square Error of the resulting clustering on the training data

        Page 45 of KMeansClusteringMLSummer2020.podf

        NOTE: BIG-F***ING NOTE:
        The assigned reading uses sum square error rather than mean square error
        """

        mse = 0

        for i in range(self.k):
            # YEAH - DON'T USE SUM!!!!
            # mse += np.sum(self.euclidean_dist(centroids[i], data[assign == i, :-1]) ** 2)
            mse += np.mean(self.euclidean_dist(centroids[i], data[assign == i, :-1]) ** 2)
        return mse

    def find_sss(self, centroids: NDArray[float]) -> float:
        """ Mean Sum of Separation
        Find the SSS of the centroids.

        Page 48 of KMeanseClusteringMLSummer2020.pdf

        mss(C) = all distinct paris of clusters i, j in C (i != j)
                 --------------------------------------------------
                         K(K - 1) / 2
        """

        sss = 0.0

        for i in range(self.k):
            for j in range(self.k):
                if i != j:
                    sss = sss + (centroids[i] - centroids[j]) ** 2

        sss = np.sum(sss)

        mss = sss / ((self.k * (self.k - 1)) / 2)

        return mss / 2

    # def find_sss(self, centroids: NDArray[float]) -> float:
    #     """ Mean Sum of Separation
    #     Find the SSS of the centroids.
    #     """
    #
    #     sss = 0.0
    #
    #     for i in range(self.k):
    #         for j in range(self.k):
    #             sss = sss + (centroids[i] - centroids[j]) ** 2
    #
    #     sss = np.sum(sss)
    #
    #     return sss

    def find_entropy(self, data: NDArray[float], assignment: NDArray[float]) -> float:
        """ Find the Mean Entropy
        Find the Entropy of each cluster
        """

        ratios = np.zeros(10)
        entropy = np.zeros(self.k)
        m_entropy = 0

        # https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log
        # np.seterr(divide = 'ignore')
        # np.seterr(divide='warn')

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
