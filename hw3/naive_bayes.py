# CS 545 Machine Learning
# Summer 2020 Portland State University
# Professor: Dr. Paul Doliotis
# Steve Braich
# HW #3: Naive Bayes Classifier based on Gaussian

from recordclass import recordclass
import sys
import math
import statistics

MINIMUM_VARIANCE = 0.0001


class NaiveBayesClassifier:
    """ Naive Bayes Classifier Object
    This is class object holds the data structures and methods required to
    classify data according to Gaussian Naive Bayes.
    """

    Classifier = recordclass('Classifier', 'class_id probability attributes')
    Attribute = recordclass('Attribute', 'attr_id mean std_dev probability values')
    Sample = recordclass('Sample', 'id class_prediction probability class_label ' +
                         'accuracy probability_x_given_class probability_x')

    train_data: []
    test_data: []
    classes: []
    samples: []

    def __init__(self, train_filename: str, test_filename: str) -> None:
        """ Constructor for Naive Bayes Classifier object
        Initialize training and test data and the class and sample lists.
        """
        self.train_data = self.load(train_filename)
        self.test_data = self.load(test_filename)
        self.classes = []
        self.samples = []

    @staticmethod
    def load(file: str = None) -> []:
        """ Load data from file
        """
        with open(file) as fp:
            data = [line.rstrip('\n') for line in fp]
        return data

    @staticmethod
    def std_dev(val: [], mean: float) -> float:
        """ Get the standard deviation """
        diff = []

        for i in val:
            diff.append((i - mean) ** 2)

        var = (1 / (len(diff) - 1)) * sum(diff)
        if var < MINIMUM_VARIANCE:
            var = MINIMUM_VARIANCE

        return math.sqrt(var)

    @staticmethod
    def gaussian(x: float, mean: float, std_dev: float) -> float:
        """ Calculate Gaussian """
        a = 1 / (std_dev * math.sqrt(2 * math.pi))
        b = math.exp(-1 * (((x - mean) ** 2) / (2 * (std_dev ** 2))))

        # return np.exp(-np.power((x - mean) / stdev, 2.) / 2.)
        return a * b

    @staticmethod
    def accuracy(class_probabilities: [], actual_class: int) -> float:
        """ Get the accuracy
        Given the class probabilities and the true class labels,
        calculate the accuracy of our classifier.
        """
        maximums = []

        for i, prob in enumerate(class_probabilities):
            if prob == max(class_probabilities):
                maximums.append(i + 1)

        if len(maximums) == 1:
            if maximums[0] == actual_class:
                return 1
            else:
                return 0
        else:
            if actual_class in maximums:
                return 1 / len(maximums)
            else:
                return 0

    def train(self, data: []) -> None:
        """ Train a Gaussian Naive Bayes Classifier
        """
        class_ids = []

        # Get the number of classes
        for sample in data:
            fields = sample.split()
            fields = [float(x) for x in fields]

            if not (fields[-1] in class_ids):
                class_ids.append(fields[-1])

        # Create classifiers
        for i in range(0, len(class_ids)):
            c = self.Classifier(class_id=i + 1, probability=float(), attributes=[])
            self.classes.append(c)

            for j in range(0, len(data[0].split()) - 1):
                a = self.Attribute(attr_id=j + 1, mean=float(), std_dev=float(), probability=float, values=[])
                self.classes[i].attributes.append(a)

        # Calculate probabilities
        for sample in data:
            fields = sample.split()
            fields = [float(x) for x in fields]
            class_id = int(fields[-1])

            for c in self.classes:
                if c.class_id == class_id:
                    for index, k in enumerate(fields[:-1]):
                        c.attributes[index].values.append(k)
                c.probability = len(c.attributes[0].values) / len(data)

        # Get the mean and std dev
        for i in self.classes:
            for j in i.attributes:
                if not (len(j.values) == 0):
                    j.mean = statistics.mean(j.values)
                j.std_dev = self.std_dev(j.values, j.mean)

    def classify(self, data: []) -> float:
        """ Classify data based on trained Naive Bayes"""

        for index, i in enumerate(data):
            fields = i.split()
            fields = [float(x) for x in fields]
            sample = self.Sample(id=index + 1, class_prediction=int(), probability=float(), class_label=fields[-1],
                                 accuracy=float(), probability_x_given_class=[], probability_x=0)

            # Calculate the gaussian
            for j in range(0, len(self.classes)):
                probability_x_given_class = 1

                for ind, k in enumerate(fields[:-1]):
                    att = self.classes[j].attributes[ind]
                    probability_x_given_class *= self.gaussian(k, att.mean, att.std_dev)

                sample.probability_x_given_class.append(probability_x_given_class)
                sample.probability_x += (sample.probability_x_given_class[j] * self.classes[j].probability)

            # Calculate P(class | x) using Naive Bayes
            class_probabilities = []

            for j in range(0, len(self.classes)):
                class_probabilities.append((sample.probability_x_given_class[j] * self.classes[j].probability)
                                           / sample.probability_x)

            # Select the class with the highest probability
            sample.probability = max(class_probabilities)
            sample.class_prediction = class_probabilities.index(max(class_probabilities)) + 1
            sample.accuracy = self.accuracy(class_probabilities, sample.class_label)

            print(f"ID={sample.id:5}, predicted={sample.class_prediction:3}, probability={sample.probability:.4f}, "
                  f"true={sample.class_label:3}, accuracy={sample.class_label:4.2f}")

            self.samples.append(sample)

        total_accuracy = 0
        for i in self.samples:
            total_accuracy += i.accuracy

        return total_accuracy

    def report(self, class_accuracy: float) -> None:
        """ Report
        Print the results of the classification
        """
        for c in self.classes:
            for a in c.attributes:
                print(f"Class {c.class_id}, attribute {a.attr_id}, mean = {a.mean:.2f}, std = {a.std_dev:.2f}")
            print()

        class_accuracy = (class_accuracy / len(self.samples))

        print()
        print(f"classification accuracy = {class_accuracy:.4f}")


if __name__ == "__main__":

    # Check Command line args
    if len(sys.argv) != 3:
        print("Incorrect number of command line arguments.\n")
        print("Usage: python naive_bayes.py train_file_path test_file_path")
        sys.exit(0)

    # Instantiate a Naive Bayes Classifier object, train it, classify test data and report results
    nb = NaiveBayesClassifier(train_filename=sys.argv[1], test_filename=sys.argv[2])
    nb.train(nb.train_data)
    accuracy = nb.classify(nb.test_data)
    nb.report(accuracy)
