from unittest import TestCase
import numpy as np
import unittest.mock as mock
import sys
from mock import mock_open, patch
#from unittest.mock import mock_open
import builtins
import os

import perceptron as pt
import test_perceptron_data as testdata


class TestPerceptron(TestCase):

    def create_test_data(self, filename):
        """ Helper function
        User to create small segments of data for unit testing
        """

        datafile = np.loadtxt(filename, delimiter=',')
        dataset = np.insert(datafile[:, np.arange(1, self.input_size)] / 255, 0, self.bias, axis=1)
        data_labels = datafile[:, 0]

        np.set_printoptions(threshold=sys.maxsize, suppress=True)
        datafile_str = np.array_repr(datafile, max_line_width=None, precision=0).replace('\n', '').replace(' ', '').replace('],', '],\n')
        dataset_str = np.array_repr(dataset, max_line_width=None, precision=0).replace('\n', '').replace(' ','').replace('],', '],\n')
        data_labels_str = np.array_repr(data_labels, max_line_width=None, precision=0).replace('\n', '').replace(' ','').replace('],', '],\n')

        print(f"{filename}: datafile = np.{datafile_str}")
        print(f"{filename}: dataset = np.{dataset_str}")
        print(f"{filename}: datalabels = np.{data_labels_str}")

        #conf_maxtrix_str = np.array_repr(conf_matrix, max_line_width=None, precision=0).replace('\n', '').replace(' ','').replace('],', '],\n')


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

        assert(p.weights.shape == (785, 10))

    def test_evaluate(self):
        bias = 1
        p = pt.Perceptron(sizes=[785, 10], bias=bias)
        p.test_data = testdata.test_data_10
        p.test_labels = testdata.test_labels_10
        p.weights = testdata.trained_weights

        accuracy, predictions = p.evaluate(dataset=p.test_data, data_labels=p.test_labels)

        expected_predictions = [7, 2, 1, 0, 9, 1, 4, 9, 6, 9]

        assert(accuracy > .70)
        assert(predictions == expected_predictions)

    def test_report(self):
        bias = 1
        p = pt.Perceptron(sizes=[785, 10], bias=bias)
        p.rate = 0.1
        p.test_labels = testdata.test_labels_10

        predictions = [7, 2, 1, 0, 9, 1, 4, 9, 6, 9]
        test_accuracy = 0.80
        train_epoch_accuracy = [0.08735, 0.8756, 0.86975, 0.8623666666666666, 0.8714333333333333, 0.8639333333333333, 0.8916666666666667, 0.8642833333333333, 0.8893333333333333, 0.8867333333333334, 0.8793166666666666, 0.8778666666666667, 0.87285, 0.8519, 0.8774333333333333, 0.8725333333333334, 0.8682333333333333, 0.8848666666666667, 0.8672333333333333, 0.88585, 0.8916833333333334, 0.8859833333333333, 0.8844, 0.86395, 0.8768666666666667, 0.87785, 0.8990166666666667, 0.8808166666666667, 0.8971833333333333, 0.8925166666666666, 0.8995666666666666, 0.8684833333333334, 0.8915333333333333, 0.87115, 0.87725, 0.88015, 0.8810666666666667, 0.88425, 0.89875, 0.9054666666666666, 0.8758333333333334, 0.87295, 0.8893333333333333, 0.8859666666666667, 0.8913666666666666, 0.8685666666666667, 0.9002333333333333, 0.8891333333333333, 0.8698666666666667, 0.8714833333333334, 0.8774]
        test_epoch_accuracy = [0.0863, 0.877, 0.8631, 0.8585, 0.8631, 0.8531, 0.8861, 0.8558, 0.8825, 0.8776, 0.8684, 0.8656, 0.8702, 0.8418, 0.8654, 0.8673, 0.8595, 0.8746, 0.8555, 0.8757, 0.8786, 0.878, 0.8743, 0.8567, 0.8629, 0.8659, 0.8898, 0.8695, 0.8864, 0.8797, 0.8859, 0.8549, 0.8815, 0.8624, 0.8625, 0.8698, 0.8699, 0.8727, 0.8863, 0.8939, 0.8599, 0.8649, 0.879, 0.8732, 0.8805, 0.8561, 0.8859, 0.876, 0.8581, 0.8615, 0.8651]

        conf_matrix = p.report(rate=p.rate, prediction=predictions, test_accuracy=test_accuracy,
                 train_epoch_accuracy=train_epoch_accuracy, test_epoch_accuracy=test_epoch_accuracy)

        # @formatter:off
        expected_conf_matrix = np.array \
        ([
            [1,0,0,0,0,0,0,0,0,0],
            [0,2,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,1],
            [0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,2]
        ])

        #ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        assert(np.allclose(conf_matrix, expected_conf_matrix))

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
