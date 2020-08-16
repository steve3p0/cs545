# CS 545 Machine Learning
# Summer 2020 Portland State University
# Professor: Dr. Paul Doliotis
# Steve Braich
# HW #3: Naive Bayes Classifier based on Gaussian

import sys
import collections
from recordclass import recordclass
import statistics as stat
import math


class Attribute:
    def __init__(self, attID):
        self.attributeID = attID
        self.mean = float()
        self.stdDev = float()
        self.prob = float()
        self.values = []


class testObject:
    def __init__(self, id, trueClass):
        self.id = id
        self.pClass = int()
        self.prob = float()
        self.tClass = trueClass
        self.accuracy = float()
        self.p_xGivenCs = []
        self.p_x = 0


class NaiveBayesClassifier:

    #Classifier = collections.namedtuple('Classifier', 'classID prob attributes')
    #Attribute = collections.namedtuple('Attribute', 'attributeID mean stdDev prob values')

    Classifier = recordclass('Classifier', 'classID prob attributes')
    #Attribute = recordclass('Attribute', 'attributeID mean stdDev prob values')

    # a = Attribute(mean=1.35, stdDev=.75, prob=0.57, values=[1, 3, 5])
    # c = Classifier(prop=0.57, attributes=[a, a, a])

    # car1 = \
    # {
    #     'color': 'red',
    #     'mileage': 3812.4,
    #     'automatic': True,
    # }

    train_data: []
    test_data: []

    classes: []
    testObjs: []

    def __init__(self, train_filename: str=None, test_filename: str=None):
        self.train_data = self.load(train_filename)
        self.test_data = self.load(test_filename)
        self.classes = []
        self.testObjs = []

    def load(self, file):
        #return [line.rstrip('\n') for line in open(file)]
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

    def train(self):
        ### Training Phase ###
        classNums = []

        # Determine # of classes
        for i in self.train_data:
            tempStr = i.split()
            temp = [float(x) for x in tempStr]

            if (not (temp[-1] in classNums)):
                classNums.append(temp[-1])

        # Create class objects and associated attribute objects
        for i in range(0, len(classNums)):
            #self.classes.append(Classifier(i + 1))
            c = self.Classifier(classID=i+1, prob=float(), attributes=[])
            self.classes.append(c)

            for j in range(0, len(self.train_data[0].split()) - 1):
                self.classes[i].attributes.append(Attribute(j + 1))

        # Find and file values to associated attribute object
        for i in self.train_data:
            tempStr = i.split()
            temp = [float(x) for x in tempStr]
            clNum = int(temp[-1])

            for j in self.classes:
                if (j.classID == clNum):
                    for index, k in enumerate(temp[:-1]):
                        j.attributes[index].values.append(k)

        # Calculate p(C)
        for k in self.classes:
            k.prob = len(k.attributes[0].values) / len(self.train_data)
            #k.prob._replace()

        # Calulate mean and standard deviation for each attribute
        for i in self.classes:
            for j in i.attributes:
                if (not (len(j.values) == 0)):
                    j.mean = stat.mean(j.values)
                j.stdDev = self.find_stdDev(j.values, j.mean)

    def classify(self):
        for index, i in enumerate(self.test_data):
            tempStr = i.split()
            temp = [float(x) for x in tempStr]
            tempObj = testObject(index + 1, temp[-1])

            # Calculate gaussians for each class on the test object
            for j in range(0, len(self.classes)):
                p_xGivenC = 1

                for ind, k in enumerate(temp[:-1]):
                    tempAtt = self.classes[j].attributes[ind]
                    p_xGivenC *= self.calc_gaussian(k, tempAtt.mean, tempAtt.stdDev)

                tempObj.p_xGivenCs.append(p_xGivenC)

            # Calculate p(x) with sum rule
            for j in range(0, len(self.classes)):
                tempObj.p_x += (tempObj.p_xGivenCs[j] * self.classes[j].prob)

            # Use Bayes Rule to calculate P(C|x)
            classProbs = []

            for j in range(0, len(self.classes)):
                classProbs.append((tempObj.p_xGivenCs[j] * self.classes[j].prob) / tempObj.p_x)

            # Take the max value as the identified class
            tempObj.prob = max(classProbs)
            tempObj.pClass = classProbs.index(max(classProbs)) + 1
            tempObj.accuracy = self.getAccuracy(classProbs, tempObj.tClass)
            print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (
            tempObj.id, tempObj.pClass, tempObj.prob, tempObj.tClass, tempObj.accuracy))

            self.testObjs.append(tempObj)

        # Calculate total accuracy
        accuracySum = 0
        for i in self.testObjs:
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

        print("\nclassification accuracy = %6.4f" % (accuracySum / len(self.testObjs)))


if __name__ == "__main__":
    assert(len(sys.argv) >= 3, "Not enough command line arguments!")
    nb = NaiveBayesClassifier(sys.argv[1], sys.argv[2])
    nb.train()
    accuracy = nb.classify()
    nb.report(accuracy)