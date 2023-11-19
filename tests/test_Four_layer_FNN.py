import unittest
from ParametricSpectralClustering import Four_layer_FNN
import torch
import numpy as np

m = Four_layer_FNN(64, 128, 256, 64, 10)


class test_Four_layer_FNN(unittest.TestCase):
    def test_init(self):
        self.assertIs(type(m.hidden1), torch.nn.Linear)
        self.assertIs(type(m.hidden2), torch.nn.Linear)
        self.assertIs(type(m.hidden3), torch.nn.Linear)
        self.assertIs(type(m.predict), torch.nn.Linear)

    @torch.no_grad()
    def test_output_shape(self):
        x = torch.randn(1797, 64)
        outputs = m(x)
        self.assertEqual(10, outputs.shape[1])


if __name__ == "__main__":
    unittest.main()

# python -m unittest -v test_Four_layer_FNN.test_Four_layer_FNN
