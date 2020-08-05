import unittest
import numpy as np
import unittest.mock as mock
import sys

import network as nn
import test_network_data as testdata


class TestNetworkExperiments(unittest.TestCase):
    """ Integration Tests for HW #2 Experiments

    You can run all of the experiments by running the following test:

        test_all_experiments()

    Or, you can run the three experiments separately:

        test_experiment1()
        test_experiment2()
        test_experiment3()

    ASSUMPTION:
    The MNIST train and test files in current directory with exact filename below.

    Modify the train_file, test_file, epoch settings below
    """

    # Settings for a full blown integration test
    epochs = 50
    expected_accuracy = .90
    train_file = 'mnist_train.csv'        # Full size training file
    test_file = 'mnist_validation.csv'    # Full size training file

    # Settings for a veru small integration test
    # epochs = 3
    # expected_accuracy = .80
    # train_file = 'mnist_train_6k.csv'       # Small train file for quick test
    # test_file = 'mnist_validation_1k.csv'   # Small test  file for quick test

    def test_all_experiments(self):
        """ Full integration test of all experiments
        """

        # Initial dimensions of the layers
        input_size = 785
        hidden_size = 20
        output_size = 10
        sizes = [input_size, hidden_size, output_size]

        n = nn.Network(sizes, train_filename=self.train_file, test_filename=self.test_file)

        print(f"\n\n#### Experiment 1: Hidden Layer Size #####################")

        print(f"\nExperiment #1: {hidden_size} hidden nodes")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        hidden_size = 50
        sizes = [input_size, hidden_size, output_size]
        n.resize(sizes)
        print(f"\nExperiment #1: {hidden_size} hidden nodes")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        hidden_size = 100
        sizes = [input_size, hidden_size, output_size]
        n.resize(sizes)
        print(f"\nExperiment #1: {hidden_size} hidden nodes")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        print(f"\n\n#### Experiment 3: Momentum ############################")

        momentum = 0.25
        print(f"\nExperiment #3: {momentum} momentum")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(α=momentum, epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        momentum = 0.5
        print(f"\nExperiment #3: {momentum} momentum")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(α=momentum, epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        momentum = 0.95
        print(f"\nExperiment #3: {momentum} momentum")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(α=momentum, epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        print(f"\n\n#### Experiment 2: Training Size #######################")
        total_training_samples = len(n.train_labels)

        training_size = 0.50
        samples = int(total_training_samples * training_size)
        print(f"\nExperiment #2: {training_size} training data")
        print(f"--------------------------------------------------")
        n.train_labels = n.train_labels[:samples]
        n.train_data = n.train_data[:samples]
        wᵢ, wⱼ, accuracy = n.train(epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        training_size = 0.25
        samples = int(total_training_samples * training_size)
        print(f"\nExperiment #2: {training_size} training data")
        print(f"--------------------------------------------------")
        n.train_labels = n.train_labels[:samples]
        n.train_data = n.train_data[:samples]
        wᵢ, wⱼ, accuracy = n.train(epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

    def test_experiment1(self):
        """ Full integration test of Experiment #1
        """

        # Initial dimensions of the layers
        input_size = 785
        hidden_size = 20
        output_size = 10
        sizes = [input_size, hidden_size, output_size]

        n = nn.Network(sizes, train_filename=self.train_file, test_filename=self.test_file)

        print(f"\n\n#### Experiment 1: Hidden Layer Size #####################")

        print(f"\nExperiment #1: {hidden_size} hidden nodes")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        hidden_size = 50
        sizes = [input_size, hidden_size, output_size]
        n.resize(sizes)
        print(f"\nExperiment #1: {hidden_size} hidden nodes")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        hidden_size = 100
        sizes = [input_size, hidden_size, output_size]
        n.resize(sizes)
        print(f"\nExperiment #1: {hidden_size} hidden nodes")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

    def test_experiment2(self):
        """ Full integration test of Experiment #2
        """

        # Initial dimensions of the layers
        input_size = 785
        hidden_size = 100
        output_size = 10
        sizes = [input_size, hidden_size, output_size]

        n = nn.Network(sizes, train_filename=self.train_file, test_filename=self.test_file)

        print(f"\n\n#### Experiment 2: Training Size #######################")
        total_training_samples = len(n.train_labels)

        training_size = 0.50
        samples = int(total_training_samples * training_size)
        print(f"\nExperiment #2: {training_size} training data")
        print(f"--------------------------------------------------")
        n.train_labels = n.train_labels[:samples]
        n.train_data = n.train_data[:samples]
        wᵢ, wⱼ, accuracy = n.train(epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        training_size = 0.25
        samples = int(total_training_samples * training_size)
        print(f"\nExperiment #2: {training_size} training data")
        print(f"--------------------------------------------------")
        n.train_labels = n.train_labels[:samples]
        n.train_data = n.train_data[:samples]
        wᵢ, wⱼ, accuracy = n.train(epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

    def test_experiment3(self):
        """ Full integration test of all experiments
        """

        # Initial dimensions of the layers
        input_size = 785
        hidden_size = 100
        output_size = 10
        sizes = [input_size, hidden_size, output_size]

        n = nn.Network(sizes, train_filename=self.train_file, test_filename=self.test_file)

        print(f"\n\n#### Experiment 3: Momentum ############################")

        momentum = 0.25
        print(f"\nExperiment #3: {momentum} momentum")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(α=momentum, epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        momentum = 0.5
        print(f"\nExperiment #3: {momentum} momentum")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(α=momentum, epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

        momentum = 0.95
        print(f"\nExperiment #3: {momentum} momentum")
        print(f"--------------------------------------------------")
        wᵢ, wⱼ, accuracy = n.train(α=momentum, epochs=self.epochs)
        assert(accuracy > self.expected_accuracy)

    def test_experiment1_hidden20(self):
        """ INTEGRATION TEST fpr Experiment #1 (20 Hidden Nodes)

        ASSUMPTION:
        That you have the MNIST train and test files in current directory
        with exact filename below
        """

        # train_file = 'mnist_train.csv'
        # test_file = 'mnist_validation.csv'
        train_file = 'mnist_train_6k.csv'
        test_file = 'mnist_validation_1k.csv'
        # train_file = 'mnist_train_15k.csv'
        # test_file = 'mnist_validation_1k.csv'

        epochs = 3                  # 3, 10, 50
        expected_accuracy = .80     # .80 for small tests, .90 for full test

        input_size = 785
        hidden_size = 20
        output_size = 10
        sizes = [input_size, hidden_size, output_size]

        n = nn.Network(sizes, train_filename=train_file, test_filename=test_file)

        wᵢ, wⱼ, accuracy = n.train(epochs=epochs)
        assert(accuracy > expected_accuracy)


class TestNetwork(unittest.TestCase):

    @unittest.skip("Helper function")
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

        # conf_maxtrix_str = np.array_repr(conf_matrix, max_line_width=None, precision=0).replace('\n', '').replace(' ','').replace('],', '],\n')

    def test__init__nofileload(self):
        layers = 3
        bias = 1

        input_size = 785
        hidden_size = 20
        output_size = 10
        sizes = [input_size, hidden_size, output_size]
        n = nn.Network(sizes=sizes)

        assert(n.layers == layers)
        assert(n.sizes == sizes)
        assert(n.input_size == input_size)
        assert(n.hidden_size == hidden_size)
        assert(n.output_size == output_size)

        assert(n.bias == bias)

        assert(n.η == 0.0)
        assert(n.α == 0.0)
        assert(n.epochs == 0)

        assert(n.wᵢ.shape == (input_size, hidden_size))
        assert(n.wⱼ.shape == (hidden_size + bias, output_size))

        assert(n.train_data is None)
        assert(n.train_labels is None)
        assert(n.test_data is None)
        assert(n.test_labels is None)

    @mock.patch('numpy.loadtxt')
    def test__init__with_fileload(self, np_loadtxt):
        train_file = 'MOCK FILE PATH'
        test_file = 'MOCK FILE PATH'
        np_loadtxt.return_value = testdata.test_datafile_10

        layers = 3
        input_size = 785
        hidden_size = 20
        output_size = 10
        sizes = [input_size, hidden_size, output_size]
        bias = 1

        n = nn.Network(sizes=sizes, train_filename=train_file, test_filename=test_file)

        assert(n.layers == layers)
        assert(n.sizes == sizes)
        assert(n.input_size == input_size)
        assert(n.hidden_size == hidden_size)
        assert(n.output_size == output_size)

        assert(n.bias == bias)

        assert(n.η == 0.0)
        assert(n.α == 0.0)
        assert(n.epochs == 0)

        assert(n.wᵢ.shape == (input_size, hidden_size))
        assert(n.wⱼ.shape == (hidden_size + bias, output_size))

        assert(np.allclose(n.test_data, testdata.test_data_10))
        assert(np.allclose(n.test_labels, testdata.test_labels_10))
        assert(np.allclose(n.train_data, testdata.test_data_10))
        assert(np.allclose(n.train_labels, testdata.test_labels_10))

    @mock.patch('numpy.loadtxt')
    def test_load(self, np_loadtxt):
        np_loadtxt.return_value = testdata.train_datafile_60

        input_size = 785
        hidden_size = 20
        output_size = 10
        sizes = [input_size, hidden_size, output_size]

        filename = "non-existant path"
        n = nn.Network(sizes=sizes)

        data, labels = n.load(filename=filename)

        assert(data.shape == (60, 785))
        assert(labels.shape == (60, ))

        assert(np.allclose(data, testdata.train_data_60))
        assert(np.allclose(labels, testdata.train_labels_60))

    def test_forward(self):
        bias = 1
        initial_weight_low = -0.05
        initial_weight_high = 0.05

        input_size = 785
        hidden_size = 20
        output_size = 10
        sizes = [input_size, hidden_size, output_size]
        n = nn.Network(sizes=sizes)

        # TODO: hardcode these instead of randomizing - so that we can test expected output
        n.wᵢ = np.random.uniform(low=initial_weight_low, high=initial_weight_high, size=(input_size, hidden_size))
        n.wⱼ = np.random.uniform(low=initial_weight_low, high=initial_weight_high, size=(hidden_size + 1, output_size))

        xᵢ = testdata.train_data_60[0]
        hⱼ = np.ones(hidden_size + 1)

        # TODO: What are you doing with hⱼ_out?
        hⱼ_out, oₖ = n.forward(xᵢ, hⱼ)
        assert (oₖ.shape == (10, ))
        # TODO: Test specific values of oₖ

    def test_back(self):

        # TO DO:
        # YOU NEED TO MOCK A LOT OF THIS SHIT

        rate = 0.1
        momentum = 0.9
        target = 0.9
        initial_weight_low = -0.05
        initial_weight_high = 0.05

        input_size = 785
        hidden_size = 20
        output_size = 10
        sizes = [input_size, hidden_size, output_size]
        n = nn.Network(sizes=sizes)

        n.η = rate
        n.α = momentum

        n.wᵢ = np.random.uniform(low=initial_weight_low, high=initial_weight_high, size=(input_size, hidden_size))
        n.wⱼ = np.random.uniform(low=initial_weight_low, high=initial_weight_high, size=(hidden_size + 1, output_size))
        n.Δwⱼᵢ = np.zeros(n.Δwⱼᵢ.shape)
        n.Δwₖⱼ = np.zeros(n.Δwₖⱼ.shape)

        n.train_labels = testdata.train_labels_60

        k = 0
        xᵢ = testdata.train_data_60[k]
        hⱼ = np.ones(hidden_size + 1)

        # TODO: MOCK THIS!!!
        hⱼ, oₖ = n.forward(xᵢ=xᵢ, hⱼ=hⱼ)
        tₖ = np.ones((output_size, output_size), float) - target

        # TODO: MOCK THIS!!!
        label = int(n.train_labels[0])  # YOU PICKED THE ZERO index (first training label)
        δₖ = oₖ * (1 - oₖ) * (tₖ[label] - oₖ)
        δⱼ = hⱼ * (1 - hⱼ) * (np.dot(n.wⱼ, δₖ))

        n.back(xᵢ=xᵢ, hⱼ=hⱼ, δⱼ=δⱼ, δₖ=δₖ)

        # Test Shape of weightss
        assert(n.wᵢ.shape == (input_size, hidden_size))
        assert(n.wⱼ.shape == (hidden_size + 1, output_size))

    def test_learn(self):

        input_size = 785
        hidden_size = 20
        output_size = 10
        sizes = [input_size, hidden_size, output_size]

        rate = 0.1
        momentum = 0.9
        target = 0.9
        initial_weight_low = -0.05
        initial_weight_high = 0.05

        n = nn.Network(sizes=sizes)

        n.η = rate
        n.α = momentum

        n.train_labels = testdata.train_labels_60
        n.train_data = testdata.train_data_60
        n.wᵢ = np.random.uniform(low=initial_weight_low, high=initial_weight_high, size=(input_size, hidden_size))
        n.wⱼ = np.random.uniform(low=initial_weight_low, high=initial_weight_high, size=(hidden_size + 1, output_size))
        n.Δwⱼᵢ = np.zeros(n.Δwⱼᵢ.shape)
        n.Δwₖⱼ = np.zeros(n.Δwₖⱼ.shape)

        tₖ = np.ones((output_size, output_size), float) - target
        np.fill_diagonal(tₖ, target)
        hⱼ = np.ones(hidden_size + 1)

        n.learn(hⱼ=hⱼ, tₖ=tₖ)

        # Test Shape of weights from input to hidden
        assert(n.wᵢ.shape == (input_size, hidden_size))
        # Test Shape of weights from hidden to output
        # TODO: is the shape of wⱼ : hidden + 1, ....?
        assert(n.wⱼ.shape == (hidden_size + 1, output_size))

    def test_evaluate(self):

        initial_weight_low = -0.05
        initial_weight_high = 0.05

        input_size = 785
        hidden_size = 100
        output_size = 10
        sizes = [input_size, hidden_size, output_size]
        n = nn.Network(sizes=sizes)

        n.test_data = testdata.test_data_10
        n.test_labels = testdata.test_labels_10

        # n.wᵢ = np.random.uniform(low=initial_weight_low, high=initial_weight_high, size=(input_size, hidden_size))
        # n.wⱼ = np.random.uniform(low=initial_weight_low, high=initial_weight_high, size=(hidden_size + 1, output_size))

        n.wᵢ = testdata.wi_experiment2_halftrain
        n.wⱼ = testdata.wj_experiment2_halftrain
        accuracy, predictions = n.evaluate(dataset=n.test_data, data_labels=n.test_labels)

        expected_predictions = [7, 2, 1, 0, 9, 1, 4, 9, 6, 9]

        assert(accuracy > .70)

    def test_report(self):
        input_size = 785
        hidden_size = 20
        output_size = 10
        sizes = [input_size, hidden_size, output_size]
        n = nn.Network(sizes=sizes)

        n.rate = 0.1
        n.test_labels = testdata.test_labels_10

        predictions = [7, 2, 1, 0, 9, 1, 4, 9, 6, 9]
        train_epoch_accuracy = [0.08735, 0.8756, 0.86975, 0.8623666666666666, 0.8714333333333333, 0.8639333333333333, 0.8916666666666667, 0.8642833333333333, 0.8893333333333333, 0.8867333333333334, 0.8793166666666666, 0.8778666666666667, 0.87285, 0.8519, 0.8774333333333333, 0.8725333333333334, 0.8682333333333333, 0.8848666666666667, 0.8672333333333333, 0.88585, 0.8916833333333334, 0.8859833333333333, 0.8844, 0.86395, 0.8768666666666667, 0.87785, 0.8990166666666667, 0.8808166666666667, 0.8971833333333333, 0.8925166666666666, 0.8995666666666666, 0.8684833333333334, 0.8915333333333333, 0.87115, 0.87725, 0.88015, 0.8810666666666667, 0.88425, 0.89875, 0.9054666666666666, 0.8758333333333334, 0.87295, 0.8893333333333333, 0.8859666666666667, 0.8913666666666666, 0.8685666666666667, 0.9002333333333333, 0.8891333333333333, 0.8698666666666667, 0.8714833333333334, 0.8774]
        test_epoch_accuracy = [0.0863, 0.877, 0.8631, 0.8585, 0.8631, 0.8531, 0.8861, 0.8558, 0.8825, 0.8776, 0.8684, 0.8656, 0.8702, 0.8418, 0.8654, 0.8673, 0.8595, 0.8746, 0.8555, 0.8757, 0.8786, 0.878, 0.8743, 0.8567, 0.8629, 0.8659, 0.8898, 0.8695, 0.8864, 0.8797, 0.8859, 0.8549, 0.8815, 0.8624, 0.8625, 0.8698, 0.8699, 0.8727, 0.8863, 0.8939, 0.8599, 0.8649, 0.879, 0.8732, 0.8805, 0.8561, 0.8859, 0.876, 0.8581, 0.8615, 0.8651]

        conf_matrix = n.report(rate=n.rate, prediction=predictions, train_epoch_accuracy=train_epoch_accuracy, test_epoch_accuracy=test_epoch_accuracy)

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

        expected_accuracy = .50

        input_size = 785
        hidden_size = 100
        output_size = 10
        sizes = [input_size, hidden_size, output_size]
        n = nn.Network(sizes=sizes)

        n.train_labels = testdata.train_labels_60
        n.train_data = testdata.train_data_60
        n.test_labels = testdata.test_labels_10
        n.test_data = testdata.test_data_10

        rate = 0.1
        momentum = 0.95
        target = 0.9
        epochs = 20

        w_i, w_j, accuracy = n.train(η=rate, α=momentum, target=target, epochs=epochs)

        # Test Shape of weights from input to hidden
        assert(w_i.shape == (input_size, hidden_size))
        # Test Shape of weights from hidden to output
        # TODO: is the shape of wⱼ : hidden + 1, ....?
        assert(w_j.shape == (hidden_size + 1, output_size))

        print(f"accuracy: {accuracy}")
        assert(accuracy > expected_accuracy)