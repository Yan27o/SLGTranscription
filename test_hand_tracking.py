import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import mediapipe as mp
import numpy as np

from main import RealTimeClassifier


class HandTrackingTests(unittest.TestCase):
    def setUp(self):
        self.classifier = RealTimeClassifier.__new__(RealTimeClassifier)
        self.classifier.mp = mp
        self.classifier.confidence_threshold = 0.8
        self.classifier.maybe_speak = Mock()
        self.classifier.predict = Mock(return_value=(0, "A", 0.95))

    @staticmethod
    def landmarks(offset=0.0):
        return [SimpleNamespace(x=0.2 + offset + (i % 5) * 0.025,
                                y=0.3 + (i // 5) * 0.04) for i in range(21)]

    def test_no_hand_skips_classification(self):
        self.classifier.detect_hand = Mock(return_value=None)
        self.classifier.annotate_frame(np.zeros((480, 640, 3), np.uint8))
        self.classifier.predict.assert_not_called()
        self.classifier.maybe_speak.assert_not_called()

    def test_crop_follows_hand_and_stays_inside_frame(self):
        left = self.classifier.hand_bounds(self.landmarks(), 640, 480)
        right = self.classifier.hand_bounds(self.landmarks(0.4), 640, 480)
        self.assertGreater(right[0], left[0])
        self.assertEqual(right[1], left[1])
        edge = self.classifier.hand_bounds(self.landmarks(0.7), 640, 480)
        self.assertEqual(edge[2], 640)
        self.assertLess(edge[0], edge[2])

    def test_classification_precedes_skeleton_drawing(self):
        self.classifier.detect_hand = Mock(return_value=self.landmarks())

        def predict(roi):
            self.assertFalse(roi.any(), "Skeleton must not enter model input")
            return 0, "A", 0.95

        self.classifier.predict.side_effect = predict
        output = self.classifier.annotate_frame(np.zeros((480, 640, 3), np.uint8))
        self.classifier.predict.assert_called_once()
        self.assertTrue(output.any())

    def test_video_timestamps_increase(self):
        self.classifier.hands = Mock()
        self.classifier.hands.detect_for_video.return_value.hand_landmarks = []
        self.classifier.last_frame_timestamp_ms = -1
        frame = np.zeros((48, 64, 3), np.uint8)
        self.classifier.detect_hand(frame)
        self.classifier.detect_hand(frame)
        calls = self.classifier.hands.detect_for_video.call_args_list
        self.assertGreater(calls[1].args[1], calls[0].args[1])


if __name__ == "__main__":
    unittest.main()
