# CS 545 Machine Learning
# Summer 2020 Portland State University
# Professor: Dr. Paul Doliotis
# Steve Braich
# HW #3: Naive Bayes Classifier based on Gaussian

import sys
import statistics as stat
import numpy as np
import math
from recordclass import recordclass


class NaiveBayesClassifier:
    """ Naive Bayes Classifier Object
    This is class object holds the data structures and methods required to
    classify data according to Gaussian Naive Bayes.
    """

    Classifier = recordclass('Classifier', 'classID prob attributes')
    Attribute = recordclass('Attribute', 'attributeID mean stdDev prob values')
    Sample = recordclass('Sample', 'id pClass prob tClass accuracy p_xGivenCs p_x')

    train_data: []
    test_data: []
    classes: []
    samples: []

    def __init__(self, train_filename: str=None, test_filename: str=None) -> None:
        """ Constructor for Naive Bayes Classifier object
        Initialize training and test data and the class and sample lists.
        """
        self.train_data = self.load(train_filename)
        self.test_data = self.load(test_filename)
        self.classes = []
        self.samples = []

    def load(self, file: str=None) -> []:
        """ Load data from file
        """
        data = ''
        with open(file) as fp:
            data = [line.rstrip('\n') for line in fp]
        return data

    def std_dev(self, val: [], mean: float) -> float:
        """ Get the standard deviation """
        diff = []

        for i in val:
            diff.append((i - mean) ** 2)

        var = ((1 / (len(diff) - 1)) * sum(diff))
        if (var < 0.0001):
            var = 0.0001

        return math.sqrt(var)

    def gaussian(self, x: float, mean: float, stdev: float) -> float:
        """ Calculate Gaussian """
        a = 1 / (stdev * math.sqrt(2 * math.pi))
        b = math.exp(-1 * (((x - mean) ** 2) / (2 * (stdev ** 2))))

        return a * b
        #return np.exp(-np.power((x - mean) / stdev, 2.) / 2.)

    def accuracy(self, class_probs: [], actual_class: int) -> float:
        """ Get the accuracy
        Given the class probabilities and the true class labels,
        calculate the accuracy of our classifier.
        """
        maximums = []

        for i, prob in enumerate(class_probs):
            if (prob == max(class_probs)):
                maximums.append(i + 1)

        if (len(maximums) == 1):
            if (maximums[0] == actual_class):
                return 1
            else:
                return 0
        else:
            if (actual_class in maximums):
                return 1 / len(maximums)
            else:
                return 0

    def train(self, data: []) -> None:
        """ Train a Gaussian Naive Bayes Classifier
        """
        class_ids = []

        # Get the number of classes
        for i in data:
            fields = i.split()
            fields = [float(x) for x in fields]

            if (not (fields[-1] in class_ids)):
                class_ids.append(fields[-1])

        # Create classifiers
        for i in range(0, len(class_ids)):
            c = self.Classifier(classID=i+1, prob=float(), attributes=[])
            self.classes.append(c)

            for j in range(0, len(data[0].split()) - 1):
                a = self.Attribute(attributeID=j+1, mean=float(), stdDev=float(), prob=float, values=[])
                self.classes[i].attributes.append(a)

        # Calculate probailities
        for i in data:
            fields = i.split()
            fields = [float(x) for x in fields]
            class_id = int(fields[-1])

            for j in self.classes:
                if (j.classID == class_id):
                    for index, k in enumerate(fields[:-1]):
                        j.attributes[index].values.append(k)
                j.prob = len(j.attributes[0].values) / len(data)

        # Get the mean and std dev
        for i in self.classes:
            for j in i.attributes:
                if (not (len(j.values) == 0)):
                    j.mean = stat.mean(j.values)
                j.stdDev = self.std_dev(j.values, j.mean)

    def classify(self, data: []) -> float:
        """ Classify data based on trained Naive Bayes"""

        for index, i in enumerate(data):
            tempStr = i.split()
            temp = [float(x) for x in tempStr]
            sample = self.Sample(id=index+1, pClass=int(), prob=float(), tClass=temp[-1], accuracy=float(), p_xGivenCs=[], p_x=0)

            # Calculate gaussians for each class on the test object
            for j in range(0, len(self.classes)):
                p_xGivenC = 1

                for ind, k in enumerate(temp[:-1]):
                    tempAtt = self.classes[j].attributes[ind]
                    p_xGivenC *= self.gaussian(k, tempAtt.mean, tempAtt.stdDev)

                sample.p_xGivenCs.append(p_xGivenC)
                sample.p_x += (sample.p_xGivenCs[j] * self.classes[j].prob)

            # Use Bayes Rule to calculate P(C|x)
            classProbs = []

            for j in range(0, len(self.classes)):
                classProbs.append((sample.p_xGivenCs[j] * self.classes[j].prob) / sample.p_x)

            # Take the max value as the identified class
            sample.prob = max(classProbs)
            sample.pClass = classProbs.index(max(classProbs)) + 1
            sample.accuracy = self.accuracy(classProbs, sample.tClass)
            print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (
            sample.id, sample.pClass, sample.prob, sample.tClass, sample.accuracy))

            self.samples.append(sample)

        # Calculate total accuracy
        accuracySum = 0
        for i in self.samples:
            accuracySum += i.accuracy

        return accuracySum

    def report(self, accuracySum: float) -> None:
        """ Report
        Print the results of the classificiation
        """
        for i in self.classes:
            for j in i.attributes:
                print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (i.classID, j.attributeID, j.mean, j.stdDev))
            print()

        print("\nclassification accuracy = %6.4f" % (accuracySum / len(self.samples)))


if __name__ == "__main__":
    assert(len(sys.argv) >= 3, "Not enough command line arguments!")
    nb = NaiveBayesClassifier(sys.argv[1], sys.argv[2])
    nb.train(nb.train_data)
    accuracy = nb.classify(nb.test_data)
    nb.report(accuracy)