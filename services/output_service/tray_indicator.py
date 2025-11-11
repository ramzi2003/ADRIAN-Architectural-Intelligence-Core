"""System tray indicator for ADRIAN output service."""

from __future__ import annotations

import logging
import threading
from typing import Dict, Tuple


logger = logging.getLogger("output-service.tray")


class TrayIndicator:
    """Manage a simple system-tray indicator that reflects voice pipeline state."""

    _STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
        "idle": (52, 152, 219),       # blue
        "listening": (46, 204, 113),  # green
        "speaking": (241, 196, 15),   # yellow
        "error": (231, 76, 60),       # red
    }

    def __init__(self, title: str = "ADRIAN"):
        self.title = title
        self.available = False
        self._icon = None
        self._state = "idle"
        self._thread: threading.Thread | None = None

        try:
            import pystray
            from PIL import Image, ImageDraw

            self._pystray = pystray
            self._Image = Image
            self._ImageDraw = ImageDraw
            image = self._create_image(self._state)
            self._icon = pystray.Icon(self.title, image, title)
            self.available = True
            logger.info("Tray indicator initialized")
        except Exception as exc:
            logger.warning(f"Tray indicator disabled: {exc}")

    def _create_image(self, state: str):
        color = self._STATE_COLORS.get(state, self._STATE_COLORS["idle"])
        image = self._Image.new("RGB", (64, 64), color)
        draw = self._ImageDraw.Draw(image)
        draw.ellipse((8, 8, 56, 56), fill=(255, 255, 255))
        draw.ellipse((16, 16, 48, 48), fill=color)
        return image

    def start(self):
        if not self.available or self._icon is None:
            return
        if self._thread and self._thread.is_alive():
            return
        try:
            self._icon.run_detached()
        except AttributeError:
            # Fallback for environments without run_detached
            self._thread = threading.Thread(target=self._icon.run, daemon=True)
            self._thread.start()

    def stop(self):
        if self._icon:
            try:
                self._icon.stop()
            except Exception as exc:
                logger.debug(f"Tray icon stop failed: {exc}")

    def set_state(self, state: str):
        if not self.available or self._icon is None:
            return
        if state not in self._STATE_COLORS:
            logger.debug(f"Unknown tray state '{state}', defaulting to idle")
            state = "idle"
        self._state = state
        try:
            self._icon.icon = self._create_image(state)
            self._icon.title = f"{self.title} - {state.capitalize()}"
        except Exception as exc:
            logger.debug(f"Failed to update tray icon: {exc}")


