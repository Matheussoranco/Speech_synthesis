import unittest
from src.utils import preprocess_text

class TestUtils(unittest.TestCase):
    def test_preprocess_text(self):
        self.assertEqual(preprocess_text(' Hello World! '), 'hello world!')

if __name__ == '__main__':
    unittest.main()
