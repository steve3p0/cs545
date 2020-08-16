from unittest import TestCase

import naive_bayes as nb


class TestNaiveBayesClassifierIntegration(TestCase):

    def test_naivebayes_pendigits(self):
        train_file = 'pendigits_training.txt'
        test_file = 'pendigits_test.txt'

        n = nb.NaiveBayesClassifier(train_filename=train_file, test_filename=test_file)
        n.train(n.train_data)
        accuracy = n.classify(n.test_data)
        n.report(accuracy)

    def test_naivebayes_satellite(self):
        train_file = 'satellite_training.txt'
        test_file = 'satellite_test.txt'

        n = nb.NaiveBayesClassifier(train_filename=train_file, test_filename=test_file)
        n.train(n.train_data)
        accuracy = n.classify(n.test_data)
        n.report(accuracy)

    def test_naivebayes_yeast(self):
        train_file = 'yeast_training.txt'
        test_file = 'yeast_test.txt'

        n = nb.NaiveBayesClassifier(train_filename=train_file, test_filename=test_file)
        n.train(n.train_data)
        accuracy = n.classify(n.test_data)
        n.report(accuracy)