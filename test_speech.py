import threading
import unittest
from unittest.mock import Mock, patch

from main import RealTimeClassifier
from speech import SpeechPlayer


class SpeechTests(unittest.TestCase):
    def test_waits_for_stability_and_respects_global_cooldown(self):
        c = RealTimeClassifier.__new__(RealTimeClassifier)
        c.tts = Mock()
        c.tts.say.return_value = True
        c.confidence_threshold = 0.8
        c.speech_candidate = None
        c.speech_candidate_since = 0.0
        c.last_spoken_time = float("-inf")
        c.speak_cooldown_sec = 1.5
        with patch("main.time.monotonic") as clock:
            clock.return_value = 10.0
            c.maybe_speak("A", 0.95)
            clock.return_value = 10.4
            c.maybe_speak("A", 0.95)
            c.tts.say.assert_not_called()
            clock.return_value = 10.6
            c.maybe_speak("A", 0.95)
            c.tts.say.assert_called_once_with("A")
            clock.return_value = 10.7
            c.maybe_speak("B", 0.95)
            clock.return_value = 11.3
            c.maybe_speak("B", 0.95)
            self.assertEqual(c.tts.say.call_count, 1)
            clock.return_value = 12.2
            c.maybe_speak("B", 0.95)
            self.assertEqual(c.tts.say.call_count, 2)
            c.maybe_speak("B", 0.2)
            self.assertIsNone(c.speech_candidate)

    def test_playback_does_not_block_or_queue_stale_letters(self):
        entered = threading.Event()
        release = threading.Event()
        engine = Mock()

        def play():
            entered.set()
            release.wait(timeout=5)

        engine.runAndWait.side_effect = play
        with patch("pyttsx3.init", return_value=engine):
            player = SpeechPlayer()
            try:
                self.assertTrue(player.ready.wait(timeout=5))
                self.assertEqual(player.status, "On", player.error)
                self.assertTrue(player.say("A"))
                self.assertTrue(entered.wait(timeout=2))
                self.assertFalse(player.say("B"))
                engine.say.assert_called_once_with("A")
            finally:
                release.set()
                player.close()
            self.assertFalse(player._thread.is_alive())
            engine.stop.assert_called_once()

    def test_initialization_failure_keeps_caller_running(self):
        with patch("pyttsx3.init", side_effect=RuntimeError("test unavailable")):
            player = SpeechPlayer()
            self.assertTrue(player.ready.wait(timeout=5))
            self.assertEqual(player.status, "Unavailable")
            self.assertFalse(player.say("A"))
            player.close()


if __name__ == "__main__":
    unittest.main()
