from unittest import TestCase
import numpy as np
import unittest.mock as mock
from mock import mock_open, patch
#from unittest.mock import mock_open
import builtins
import os

import perceptron as pt
import test_perceptron_data as testdata


class TestPerceptron(TestCase):

    def test__init__nofileload(self):
        sizes = [785, 10]
        bias = 1

        p = pt.Perceptron(sizes=sizes, bias=bias)

        assert(p.layers == 2)
        assert(p.sizes == sizes)
        assert(p.input_size == 785)
        assert(p.output_size == 10)

        assert(p.bias == bias)
        assert(p.epochs == 0)
        assert(p.rate == 0.0)
        assert(p.weights.shape == (785, 10))

        assert(p.train_data is None)
        assert(p.train_data is None)
        assert(p.train_data is None)
        assert(p.train_data is None)

    @mock.patch('numpy.loadtxt')
    def test__init__with_fileload(self, np_loadtxt):

        train_file = 'MOCK FILE PATH'
        test_file = 'MOCK FILE PATH'
        np_loadtxt.return_value = testdata.test_datafile_10
        bias = 1

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)

        assert(p.layers == 2)
        assert(p.input_size == 785)
        assert(p.output_size == 10)
        assert(p.bias == bias)

        assert(p.epochs == 0)
        assert(p.rate == 0.0)
        assert(p.weights.shape == (785, 10))

        assert(np.allclose(p.test_data, testdata.test_data_10))
        assert(np.allclose(p.test_labels, testdata.test_labels_10))
        assert(np.allclose(p.train_data, testdata.test_data_10))
        assert(np.allclose(p.train_labels, testdata.test_labels_10))

    @mock.patch('numpy.loadtxt')
    def test_load(self, np_loadtxt):
        np_loadtxt.return_value = testdata.train_datafile_60

        sizes = [785, 10]
        bias = 1
        filename = "non-existant path"
        p = pt.Perceptron(sizes=sizes, bias=bias)

        data, labels = p.load(filename=filename)

        assert (data.shape == (60, 785))
        assert (labels.shape == (60, ))

        assert(np.allclose(data, testdata.train_data_60))
        assert(np.allclose(labels, testdata.train_labels_60))

    def test_forward(self):
        bias = 1
        initial_weight_low = -0.05
        initial_weight_high = 0.05
        p = pt.Perceptron(sizes=[785, 10], bias=bias)

        p.weights = np.random.uniform(low=initial_weight_low, high=initial_weight_high,
                                         size=(p.input_size, p.output_size))
        xᴷ = testdata.train_data_60[0]

        yᴷ = p.forward(xᴷ)
        assert (yᴷ.shape == (10, ))

    def test_back(self):
        self.fail()

    def test_learn(self):
        bias = 1
        initial_weight_low = -0.05
        initial_weight_high = 0.05
        rate = 0.1

        p = pt.Perceptron(sizes=[785, 10], bias=bias)

        p.train_labels = testdata.train_labels_60
        p.train_data = testdata.train_data_60

        p.weights = np.random.uniform(low=initial_weight_low, high=initial_weight_high,
                                         size=(p.input_size, p.output_size))

        p.weights = p.learn(rate)

        assert (p.weights.shape == (785, 10))

    def test_evaluate(self):
        self.fail()

    def test_report(self):
        self.fail()

    def test_train(self):
        bias = 1
        epochs = 3
        rate = 0.1

        p = pt.Perceptron(sizes=[785, 10], bias=bias)

        p.train_labels = testdata.train_labels_60
        p.train_data = testdata.train_data_60
        p.test_labels = testdata.test_labels_10
        p.test_data = testdata.test_data_10

        model, accuracy = p.train(rate=rate, epochs=epochs)

        assert (model.shape == (785, 10))
        assert (accuracy > .40)
