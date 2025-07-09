import unittest
from ParametricSpectralClustering import PSC, Accuracy
from sklearn.datasets import load_digits
import numpy as np
from unittest.mock import patch
import io

digits = load_digits()
x = digits.data / 16
y = digits.target
psc = PSC(batch_size_data=1797)
cluster_idx = psc.fit_predict(x)
acc = Accuracy(y_true=y, y_pred=cluster_idx)


class testAccuracy(unittest.TestCase):
    def test_cluster_acc(self):
        self.assertIsInstance(acc.cluster_acc(), np.float64)

    def test_ARI(self):
        self.assertIsInstance(acc.ARI(), float)

    def test_AMI(self):
        self.assertIsInstance(acc.AMI(), (float, np.float64))


if __name__ == "__main__":
    unittest.main()
