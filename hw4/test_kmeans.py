from unittest import TestCase
import kmeans


class TestKmeans(TestCase):

    def test_experiment1(self):

        k = 10
        trainfile = 'optdigits.train'
        testfile = 'optdigits.test'

        km = kmeans.Kmeans(k=k, trainfile=trainfile, testfile=testfile)
        km.train()
        accuracy, confusion_matrix = km.evaluate(data=km.test_data)
        km.report(test_accuracy=accuracy, confusion_matrix=confusion_matrix)

        assert(accuracy > .70)

    def test_experiment2(self):

        k = 30
        trainfile = 'optdigits.train'
        testfile = 'optdigits.test'

        km = kmeans.Kmeans(k=k, trainfile=trainfile, testfile=testfile)
        km.train()
        accuracy, confusion_matrix = km.evaluate(data=km.test_data)
        km.report(test_accuracy=accuracy, confusion_matrix=confusion_matrix)

        assert(accuracy > .90)