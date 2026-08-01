import unittest
from src.utils import preprocess_text

from conftest import requires_espeak


class TestUtils(unittest.TestCase):
    @requires_espeak
    def test_preprocess_text(self):
        """preprocess_text lowercases, strips, and phonemizes.

        This previously asserted the *input* came back unchanged
        ('hello world!'), which only held because phonemization was failing
        silently and falling back to raw text. With espeak wired up the
        function returns IPA, so assert on that instead.
        """
        result = preprocess_text(' Hello World! ')
        self.assertNotEqual(result.strip(), 'hello world!',
                            "phonemizer fell back to graphemes")
        # IPA for "hello world" contains stress marks and non-ASCII phonemes.
        self.assertTrue(any(ord(ch) > 127 for ch in result),
                        f"expected IPA phonemes, got {result!r}")
        self.assertIn('w', result)


if __name__ == '__main__':
    unittest.main()
