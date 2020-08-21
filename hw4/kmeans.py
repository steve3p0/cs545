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
import matplotlib.cm as cm
import seaborn as sn
# from PIL import Image

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
    def assign_clusters(distances: NDArray[float]) -> NDArray[float]:
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
                clusters = self.assign_clusters(distances)

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
        assignments = self.assign_clusters(distance)
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
        # print("These are the corresponding labels for each cluster:", self.labels)
        print(f"Training Accuracy: {self.accuracy * 100:.2f}%")
        print(f"Testing  Accuracy: {test_accuracy * 100:.2f}%")

        print("\nTraining Metrics:")
        print(f"\tAverage Mean Square Error: {self.mean_square_error}")
        print(f"\tMean Squiare Separation:   {self.mean_square_separation}")
        print(f"\tMean Entropy: {self.mean_entropy}")

        print("\nTesting Metrics:")
        print("\nConfusion Matrix: ")
        print(confusion_matrix)
        # Confusion Matrix
        df_cm = pd.DataFrame(confusion_matrix, range(len(self.labels)), range(len(self.labels)))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 12})
        plt.title(f"K = {self.k}")
        plt.show()

        if self.k == 10:
            for i in range(self.k):
                plt.imsave('exp1_image_%i.png' % i, np.array(self.centroids[i, :]).reshape(8, 8), cmap=cm.gray)


                data = np.array(self.centroids[i, :]).reshape(8, 8)

                #data = np.asarray(self.centroids[i, :], dtype="uint8")
                #data = np.array(self.centroids[i, :]).reshape(16, 16)
                #data = np.array(self.centroids[i, :])
                #from matplotlib import pyplot as plt
                #data = np.uint8

                plt.imshow(data, interpolation='nearest', cmap='gray')
                plt.show()

                # #plt.show()
                #
                # #plt.imshow(np.array(self.centroids[i, :]), cmap='Greys')
                #
                # # im = np.random.randint(0, 255, (16, 16))
                # # I = np.dstack([im, im, im])
                # # x = 5
                # # y = 5
                # # I[x, y, :] = [1, 0, 0]
                # # plt.imshow(I, interpolation='nearest')
                # # plt.imshow(im, interpolation='nearest', cmap='Greys')
                #
                # # np.uint8
                #
                # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                # from matplotlib.figure import Figure
                #
                # fig = Figure()
                # canvas = FigureCanvas(fig)
                # ax = fig.gca()
                #
                # ax.text(0.0, 0.0, "Test", fontsize=45)
                # ax.axis('off')
                #
                # canvas.draw()  # draw the canvas, cache the renderer
                #
                # # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
                # # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
                # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        if self.k == 30:
            for i in range(self.k):
                plt.imsave('exp2_image_%i.png' % i, np.array(self.centroids[i, :]).reshape(8, 8), cmap=cm.gray)

    ####################################################################################
    # MATH SHIT ######################################################################

    @staticmethod
    def euclidean_dist(centroids: NDArray[float], data: NDArray[float]) -> NDArray[float]:
        """ Euclidean Distance
        Calculate the Eculidean distance of a data point and a centroid
        """

        row = np.sqrt(np.sum((centroids - data) ** 2, axis=1))

        return row

    #def all_euclidean_dist(self, centroids: [], data: []) -> NDArray[Any, Any]:
    def all_euclidean_dist(self, centroids: NDArray[float], data: NDArray[float]) -> NDArray[float]:
        """ Euclidean distances of all data """
        all_e_dist = np.zeros([len(data), len(centroids)])

        for i in range(len(centroids)):
            all_e_dist[:, i] = self.euclidean_dist(centroids[i, :], data[:, :-1])

        return all_e_dist

    def average_mean_square_error(self, centroids: NDArray[float], data: NDArray[float], assign) -> float:
        """ Average Mean Square Error
        Calculate the Average Mean Square Error of the resulting clustering on the training data
        """

        mse = 0

        for i in range(self.k):
            mse += np.sum(self.euclidean_dist(centroids[i], data[assign == i, :-1]) ** 2)

        return mse

    def find_sss(self, centroids: NDArray[float]) -> float:
        """ Mean Sum of Separation
        Find the SSS of the centroids.
        """

        sss = 0.0

        for i in range(self.k):
            for j in range(self.k):
                sss = sss + (centroids[i] - centroids[j]) ** 2

        sss = np.sum(sss)

        return sss

    def find_entropy(self, data: NDArray[float], assignment: NDArray[float]) -> float:
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
