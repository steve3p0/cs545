from unittest import TestCase
import perceptron as pt


class TestPerceptronIntegration(TestCase):

    # TODO: Create a setup and teardown.

    #######################################
    # Full Blown - Integration Tests
    def test_train_epoch50_rate_point00001(self):
        train_file = 'mnist_train.csv'
        test_file = 'mnist_validation.csv'
        bias = 1
        epochs = 50
        rate = 0.00001

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs)

        assert (model.shape == (785, 10))
        assert (accuracy > .80)

    def test_train_epoch50_rate_point001(self):
        train_file = 'mnist_train.csv'
        test_file = 'mnist_validation.csv'
        bias = 1
        epochs = 50
        rate = 0.001

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs)

        assert (model.shape == (785, 10))
        assert (accuracy > .80)

    def test_train_epoch50_rate_point1(self):
        train_file = 'mnist_train.csv'
        test_file = 'mnist_validation.csv'
        bias = 1
        epochs = 50
        rate = 0.1

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs)

        assert (model.shape == (785, 10))
        assert (accuracy > .80)

    ###############################################################
    ## Smaller Training Sizes Integration tests

    def test_train_60_epoch10_rate_point1_initialweights_1(self):
        train_file = 'mnist_train_60.csv'
        test_file = 'mnist_validation_10.csv'
        bias = 1
        epochs = 10
        rate = 0.1

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs, initial_weight_low=-1, initial_weight_high=1)

        assert (model.shape == (785, 10))
        assert (accuracy > .70)

    def test_train_600_epoch10_rate_point1_initialweights_1(self):
        train_file = 'mnist_train_600.csv'
        test_file = 'mnist_validation_100.csv'
        bias = 1
        epochs = 10
        rate = 0.1

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs, initial_weight_low=-1, initial_weight_high=1)

        assert (model.shape == (785, 10))
        assert (accuracy > .70)

    def test_train_600_epoch10_rate_point1(self):
        train_file = 'mnist_train_600.csv'
        test_file = 'mnist_validation_100.csv'
        bias = 1
        epochs = 10
        rate = 0.1

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs)

        assert (model.shape == (785, 10))
        assert (accuracy > .70)

    def test_train_600_epoch3_rate_point1(self):
        train_file = 'mnist_train_600.csv'
        test_file = 'mnist_validation_100.csv'
        bias = 1
        epochs = 3
        rate = 0.1

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs)

        assert (model.shape == (785, 10))
        assert (accuracy > .70)

    def test_train_6k_epoch3_rate_point1(self):
        train_file = 'mnist_train_6k.csv'
        test_file = 'mnist_validation_1k.csv'
        bias = 1
        epochs = 3
        rate = 0.1

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs)

        assert (model.shape == (785, 10))
        assert (accuracy > .70)

    def test_train_6k_epoch3_rate_point001(self):
        train_file = 'mnist_train_6k.csv'
        test_file = 'mnist_validation_1k.csv'
        bias = 1
        epochs = 3
        rate = 0.001

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs)

        assert (model.shape == (785, 10))
        assert (accuracy > .70)

    def test_train_6k_epoch3_rate_point00001(self):
        train_file = 'mnist_train_6k.csv'
        test_file = 'mnist_validation_1k.csv'
        bias = 1
        epochs = 3
        rate = 0.00001

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs)

        assert (model.shape == (785, 10))
        assert (accuracy > .70)

    ##########################################################
    # Different Weights

    def test_train_epoch50_rate_point1_initialweights_1(self):
        train_file = 'mnist_train.csv'
        test_file = 'mnist_validation.csv'
        bias = 1
        epochs = 50
        rate = 0.1

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs, initial_weight_low=-1, initial_weight_high=1)

        assert (model.shape == (785, 10))
        assert (accuracy > .80)

    def test_train_epoch50_rate_point01_initialweights_1(self):
        train_file = 'mnist_train.csv'
        test_file = 'mnist_validation.csv'
        bias = 1
        epochs = 50
        rate = 0.01

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs, initial_weight_low=-1, initial_weight_high=1)

        assert (model.shape == (785, 10))
        assert (accuracy > .80)

    def test_train_epoch50_rate_point001_initialweights_1(self):
        train_file = 'mnist_train.csv'
        test_file = 'mnist_validation.csv'
        bias = 1
        epochs = 50
        rate = 0.001

        p = pt.Perceptron(sizes=[785, 10], train_filename=train_file, test_filename=test_file, bias=bias)
        model, accuracy = p.train(rate=rate, epochs=epochs, initial_weight_low=-1, initial_weight_high=1)

        assert (model.shape == (785, 10))
        assert (accuracy > .80)
