"""
FLIR Lepton Thermal Camera Wrapper

This module provides a wrapper class for FLIR Lepton thermal cameras with
background frame capture and logging capabilities, mirroring the interface
of BosonWithTelemetry for consistent use across recording scripts.

Author: [Your Name]
Date: March 2026
"""

import time
import logging
import threading
import numpy as np
from typing import Optional, Tuple

from flirpy.camera.lepton import Lepton


class LeptonWrapper:
    """
    FLIR Lepton camera wrapper with background capture and logging support.

    Manages a background thread that continuously calls grab() on the Lepton,
    providing the same logged_images / logged_tstamps interface as
    BosonWithTelemetry so recording scripts can treat both cameras uniformly.

    Attributes:
        logged_images (list): List of captured thermal frames
        logged_tstamps (list): List of corresponding system timestamps
        enable_logging (bool): Flag to control data logging
        timestamp_offset (float): Always 0.0 — Lepton uses system time directly
    """

    def __init__(self, device: Optional[int] = None,
                 loglevel: int = logging.WARNING) -> None:
        """
        Initialize the Lepton camera and start the background capture thread.

        Args:
            device: Camera device index (default: None for auto-detection)
            loglevel: Logging level (default: logging.WARNING)
        """
        logging.basicConfig(level=loglevel)

        self._camera = Lepton(loglevel=loglevel)
        if device is not None:
            self._camera.setup_video(device)

        self.logged_images: list = []
        self.logged_tstamps: list = []
        self.enable_logging: bool = False
        self.timestamp_offset: float = 0.0

        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = threading.Event()

        self.start()

    def __del__(self) -> None:
        """Cleanup resources when object is destroyed."""
        self.close()

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background capture thread."""
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="LeptonCaptureThread",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to finish."""
        if not hasattr(self, "_running"):
            return
        self._running.clear()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logging.warning("LeptonWrapper: capture thread did not stop within timeout.")

    def close(self) -> None:
        """Stop the capture thread and release the camera."""
        self.stop()
        if hasattr(self, "_camera"):
            try:
                self._camera.release()
            except Exception as e:
                logging.warning(f"LeptonWrapper: error closing camera: {e}")

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Continuously grab frames from the Lepton in a background thread.

        grab() is a blocking call that returns when a new frame is ready,
        so the loop rate is naturally governed by the camera's frame rate
        (~8.6 Hz for Lepton 2, ~27 Hz for Lepton 3).
        """
        consecutive_failures = 0
        max_failures = 10

        while self._running.is_set():
            try:
                frame = self._camera.grab()
                if frame is not None:
                    capture_time = time.time()
                    with self._lock:
                        self._latest_frame = frame
                    self._post_capture_hook(frame, capture_time)
                    consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                logging.warning(
                    f"LeptonWrapper: frame grab failed ({consecutive_failures}/{max_failures}): {e}"
                )
                if consecutive_failures >= max_failures:
                    logging.error(
                        "LeptonWrapper: too many consecutive failures, stopping capture thread."
                    )
                    self._running.clear()
                    break

    # ------------------------------------------------------------------
    # Logging control
    # ------------------------------------------------------------------

    def _post_capture_hook(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Called from the capture thread after each successful grab.

        Args:
            frame: Captured thermal image array
            timestamp: System timestamp at time of capture
        """
        if self.enable_logging:
            self.logged_images.append(frame)
            self.logged_tstamps.append(timestamp)

    def start_logging(self) -> None:
        """Start accumulating frames and timestamps into logged_images / logged_tstamps."""
        self.enable_logging = True

    def stop_logging(self) -> None:
        """Stop accumulating frames."""
        self.enable_logging = False

    def clear_logged_data(self) -> None:
        """Clear all previously logged frames and timestamps."""
        self.logged_images.clear()
        self.logged_tstamps.clear()

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_next_image(self) -> Tuple[np.ndarray, float]:
        """
        Return the most recently captured frame and the current system time.

        Note: for accurate per-frame timestamps, use logged_tstamps after a
        logged recording session — those timestamps are recorded at capture time
        in the background thread.

        Returns:
            Tuple of (image array, timestamp)

        Raises:
            RuntimeError: If no frame has been captured yet.
        """
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            raise RuntimeError(
                "No frame available yet. The camera may still be initializing."
            )
        return frame, time.time()


if __name__ == "__main__":
    """
    Demo script for live Lepton thermal camera visualization.

    Displays thermal camera feed with inferno colormap in real-time.
    Press 'q' to quit.
    """
    import cv2

    print("Starting Lepton thermal camera live view...")
    print("Press 'q' to quit")

    lepton = None
    try:
        lepton = LeptonWrapper()
        cv2.namedWindow("Lepton Thermal Camera", cv2.WINDOW_NORMAL)

        # Wait for the first frame
        for _ in range(50):
            with lepton._lock:
                ready = lepton._latest_frame is not None
            if ready:
                break
            time.sleep(0.1)

        while True:
            image, timestamp = lepton.get_next_image()

            img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colorized = cv2.applyColorMap(img_8bit, cv2.COLORMAP_INFERNO)

            print(f"Timestamp: {timestamp:.3f}\r", end="")
            cv2.imshow("Lepton Thermal Camera", colorized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the Lepton camera is connected and accessible.")

    finally:
        if lepton is not None:
            lepton.close()
        cv2.destroyAllWindows()
        print("Camera stopped and windows closed.")
