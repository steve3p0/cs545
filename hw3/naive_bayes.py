# CS 545 Machine Learning
# Summer 2020 Portland State University
# Professor: Dr. Paul Doliotis
# Steve Braich
# HW #3: Naive Bayes Classifier based on Gaussian

import sys
import statistics as stat
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

    def __init__(self, train_filename: str=None, test_filename: str=None):
        self.train_data = self.load(train_filename)
        self.test_data = self.load(test_filename)
        self.classes = []
        self.samples = []

    def load(self, file):
        line = ''
        with open(file) as fp:
            line = [line.rstrip('\n') for line in fp]
        return line

    def find_stdDev(self, val, mean):
        sqrDiff = []

        for i in val:
            sqrDiff.append((i - mean) ** 2)

        variance = ((1 / (len(sqrDiff) - 1)) * sum(sqrDiff))
        if (variance < 0.0001):
            variance = 0.0001

        return math.sqrt(variance)

    def calc_gaussian(self, x, mean, stdev):
        a = 1 / (stdev * math.sqrt(2 * math.pi))
        b = math.exp(-1 * (((x - mean) ** 2) / (2 * (stdev ** 2))))
        return a * b

    def getAccuracy(self, classProbs, trueClass):
        maxes = []

        for i, prob in enumerate(classProbs):
            if (prob == max(classProbs)):
                maxes.append(i + 1)

        if (len(maxes) == 1):
            if (maxes[0] == trueClass):
                return 1
            else:
                return 0
        else:
            if (trueClass in maxes):
                return 1 / len(maxes)
            else:
                return 0

    def train(self, data=[]):
        """ Train a Gaussian Naive Bayes Classifier

        :param data:
        :type data:
        :return:
        :rtype:
        """
        classNums = []

        # Determine # of classes
        for i in data:
            tempStr = i.split()
            temp = [float(x) for x in tempStr]

            if (not (temp[-1] in classNums)):
                classNums.append(temp[-1])

        # Create class objects and associated attribute objects
        for i in range(0, len(classNums)):
            c = self.Classifier(classID=i+1, prob=float(), attributes=[])
            self.classes.append(c)

            # Get number of columns in the data
            for j in range(0, len(data[0].split()) - 1):
                a = self.Attribute(attributeID=j+1, mean=float(), stdDev=float(), prob=float, values=[])
                self.classes[i].attributes.append(a)

        # Find and file values to associated attribute object
        for i in data:
            tempStr = i.split()
            temp = [float(x) for x in tempStr]
            clNum = int(temp[-1])

            for j in self.classes:
                if (j.classID == clNum):
                    for index, k in enumerate(temp[:-1]):
                        j.attributes[index].values.append(k)
                j.prob = len(j.attributes[0].values) / len(data)


        # Calulate mean and standard deviation for each attribute
        for i in self.classes:
            for j in i.attributes:
                if (not (len(j.values) == 0)):
                    j.mean = stat.mean(j.values)
                j.stdDev = self.find_stdDev(j.values, j.mean)

    def classify(self, data):
        for index, i in enumerate(data):
            tempStr = i.split()
            temp = [float(x) for x in tempStr]
            sample = self.Sample(id=index+1, pClass=int(), prob=float(), tClass=temp[-1], accuracy=float(), p_xGivenCs=[], p_x=0)

            # Calculate gaussians for each class on the test object
            for j in range(0, len(self.classes)):
                p_xGivenC = 1

                for ind, k in enumerate(temp[:-1]):
                    tempAtt = self.classes[j].attributes[ind]
                    p_xGivenC *= self.calc_gaussian(k, tempAtt.mean, tempAtt.stdDev)

                sample.p_xGivenCs.append(p_xGivenC)

            # Calculate p(x) with sum rule
            for j in range(0, len(self.classes)):
                sample.p_x += (sample.p_xGivenCs[j] * self.classes[j].prob)

            # Use Bayes Rule to calculate P(C|x)
            classProbs = []

            for j in range(0, len(self.classes)):
                classProbs.append((sample.p_xGivenCs[j] * self.classes[j].prob) / sample.p_x)

            # Take the max value as the identified class
            sample.prob = max(classProbs)
            sample.pClass = classProbs.index(max(classProbs)) + 1
            sample.accuracy = self.getAccuracy(classProbs, sample.tClass)
            print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (
            sample.id, sample.pClass, sample.prob, sample.tClass, sample.accuracy))

            self.samples.append(sample)

        # Calculate total accuracy
        accuracySum = 0
        for i in self.samples:
            accuracySum += i.accuracy

        return accuracySum

    def report(self, accuracySum):
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