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

    Classifier = recordclass('Classifier', 'class_id probability attributes')
    Attribute = recordclass('Attribute', 'attr_id mean std_dev probability values')
    Sample = recordclass('Sample', 'id class_probability probability class_label accuracy probability_x_given_class probability_x')

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
            c = self.Classifier(class_id=i+1, probability=float(), attributes=[])
            self.classes.append(c)

            for j in range(0, len(data[0].split()) - 1):
                a = self.Attribute(attr_id=j+1, mean=float(), std_dev=float(), probability=float, values=[])
                self.classes[i].attributes.append(a)

        # Calculate probailities
        for i in data:
            fields = i.split()
            fields = [float(x) for x in fields]
            class_id = int(fields[-1])

            for j in self.classes:
                if (j.class_id == class_id):
                    for index, k in enumerate(fields[:-1]):
                        j.attributes[index].values.append(k)
                j.probability = len(j.attributes[0].values) / len(data)

        # Get the mean and std dev
        for i in self.classes:
            for j in i.attributes:
                if (not (len(j.values) == 0)):
                    j.mean = stat.mean(j.values)
                j.std_dev = self.std_dev(j.values, j.mean)

    def classify(self, data: []) -> float:
        """ Classify data based on trained Naive Bayes"""

        for index, i in enumerate(data):
            tempStr = i.split()
            temp = [float(x) for x in tempStr]
            sample = self.Sample(id=index+1, class_probability=int(), probability=float(), class_label=temp[-1], accuracy=float(), probability_x_given_class=[], probability_x=0)

            # Calculate gaussians for each class on the test object
            for j in range(0, len(self.classes)):
                probability_x_given_class = 1

                for ind, k in enumerate(temp[:-1]):
                    tempAtt = self.classes[j].attributes[ind]
                    probability_x_given_class *= self.gaussian(k, tempAtt.mean, tempAtt.std_dev)

                sample.probability_x_given_class.append(probability_x_given_class)
                sample.probability_x += (sample.probability_x_given_class[j] * self.classes[j].probability)

            # Use Bayes Rule to calculate P(C|x)
            class_probabilities = []

            for j in range(0, len(self.classes)):
                class_probabilities.append((sample.probability_x_given_class[j] * self.classes[j].probability) / sample.probability_x)

            # Take the max value as the identified class
            sample.probability = max(class_probabilities)
            sample.class_probability = class_probabilities.index(max(class_probabilities)) + 1
            sample.accuracy = self.accuracy(class_probabilities, sample.class_label)
            print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (
            sample.id, sample.class_probability, sample.probability, sample.class_label, sample.accuracy))

            self.samples.append(sample)

        total_accuracy = 0
        for i in self.samples:
            total_accuracy += i.accuracy

        return total_accuracy

    def report(self, accuracy: float) -> None:
        """ Report
        Print the results of the classificiation
        """
        for i in self.classes:
            for j in i.attributes:
                print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (i.class_id, j.attr_id, j.mean, j.std_dev))
            print()

        print("\nclassification accuracy = %6.4f" % (accuracy / len(self.samples)))


if __name__ == "__main__":
    assert(len(sys.argv) >= 3, "Not enough command line arguments!")
    nb = NaiveBayesClassifier(sys.argv[1], sys.argv[2])
    nb.train(nb.train_data)
    accuracy = nb.classify(nb.test_data)
    nb.report(accuracy)