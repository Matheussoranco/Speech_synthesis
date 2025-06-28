import unittest
from src.model import SimpleTTSModel
import torch

class TestModel(unittest.TestCase):
    def test_forward(self):
        model = SimpleTTSModel()
        x = torch.randint(0, 255, (2, 50))
        out = model(x)
        self.assertEqual(out.shape[0], 2)

if __name__ == '__main__':
    unittest.main()
