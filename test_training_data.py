import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from main import ALPHABET_NO_JZ, label_to_char, load_and_process_data


class TrainingDataTests(unittest.TestCase):
    def test_all_letters_survive_and_match_display_labels(self):
        letters = "ABCDEFGHIKLMNOPQRSTUVWXY"
        raw_labels = [ord(letter) - ord("A") for letter in letters]
        df = pd.DataFrame(np.full((24, 784), 255), columns=[f"pixel{i}" for i in range(784)])
        df.insert(0, "label", raw_labels)
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "data.csv"
            df.to_csv(csv, index=False)
            x_train, y_train, x_test, y_test = load_and_process_data(csv, csv)
        self.assertEqual(len(y_train), 24)
        self.assertEqual(y_train.tolist(), list(range(24)))
        self.assertEqual("".join(label_to_char(int(i)) for i in y_test), letters)
        self.assertEqual(ALPHABET_NO_JZ[-1], "Y")
        self.assertEqual(x_train.shape, (24, 28, 28, 1))
        self.assertEqual(x_train.dtype, np.float32)
        self.assertTrue(np.all(x_test == 1.0))

    def test_unknown_labels_are_rejected_instead_of_silently_dropped(self):
        df = pd.DataFrame(np.zeros((1, 784)), columns=[f"pixel{i}" for i in range(784)])
        df.insert(0, "label", [9])
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "data.csv"
            df.to_csv(csv, index=False)
            with self.assertRaises(ValueError):
                load_and_process_data(csv, csv)


if __name__ == "__main__":
    unittest.main()
