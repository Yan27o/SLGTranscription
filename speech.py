"""Offline speech on a dedicated thread; never block camera inference."""

import queue
import sys
import threading


class SpeechPlayer:
    def __init__(self):
        self.status = "Starting"
        self.error = None
        self._queue = queue.Queue(maxsize=1)
        self._stopped = threading.Event()
        self._busy = threading.Event()
        self.ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gesture-speech", daemon=True)
        self._thread.start()

    def say(self, text):
        if self.status != "On" or self._stopped.is_set() or self._busy.is_set():
            return False
        self._busy.set()
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            self._busy.clear()
            return False
        return True

    def _run(self):
        engine = None
        com = None
        try:
            if sys.platform == "win32":
                import pythoncom

                pythoncom.CoInitialize()
                com = pythoncom
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 1.0)
            self.status = "On"
            self.ready.set()
            while not self._stopped.is_set():
                try:
                    text = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if not self._stopped.is_set():
                        engine.say(text)
                        engine.runAndWait()
                finally:
                    self._busy.clear()
        except Exception as exc:
            self.error = str(exc)
            self.status = "Unavailable"
            print(f"语音播报不可用，画面识别继续运行: {exc}")
        finally:
            self.ready.set()
            try:
                if engine is not None:
                    engine.stop()
                    del engine
            finally:
                if com is not None:
                    com.CoUninitialize()

    def close(self):
        self._stopped.set()
        self._thread.join(timeout=2.0)

