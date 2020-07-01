from unittest import TestCase

import Perceptron


class TestPerceptron(TestCase):

    def test_forward(self):

        p = Perceptron.Perceptron()

        p.forward()

        #self.fail()

    def test_integration_epoch1(self):

        p = Perceptron.Perceptron()
        p.train()

