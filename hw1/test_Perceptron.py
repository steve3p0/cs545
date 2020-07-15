from unittest import TestCase
import numpy as np

import perceptron as pt


class TestPerceptron(TestCase):
    def test_load(self):
        self.fail()

    def test_cost(self):
        self.fail()

    def test_forward(self):
        self.fail()

    def test_back(self):
        self.fail()

    def test_learn(self):
        self.fail()

    def test_evaluate(self):
        self.fail()

    def test_report(self):
        self.fail()

    def test_train(self):

        train_file = 'mnist_train.csv'
        test_file = 'mnist_validation.csv'
        epochs = 3

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, epochs=epochs)



        perceptron_model = p.train(rate=0.00001)
        perceptron_model = p.train(rate=0.001)
        perceptron_model = p.train(rate=0.1)

        self.fail()
