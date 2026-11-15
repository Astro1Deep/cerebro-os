"""Minimal, bounded media capture for locally authorized Telegram requests."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Final


SCREEN_CAPTURE_BINARY: Final[str] = "/usr/sbin/screencapture"
CAPTURE_TIMEOUT_SECONDS: Final[int] = 15
MINIMUM_CAPTURE_BYTES: Final[int] = 256
CAMERA_WARMUP_SECONDS: Final[float] = 1.5
CAMERA_FRAME_INTERVAL_SECONDS: Final[float] = 0.08
MINIMUM_VISIBLE_FRAME_MAX: Final[int] = 12
MINIMUM_VISIBLE_FRAME_MEAN: Final[float] = 1.5


class RemoteCaptureError(RuntimeError):
    """Capture was unavailable or did not produce a usable image."""


def _new_capture_path(directory: Path, prefix: str) -> Path:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not directory.is_dir():
        raise RemoteCaptureError("el directorio temporal de capturas no está disponible")
    descriptor, filename = tempfile.mkstemp(prefix=prefix, suffix=".png", dir=directory)
    os.close(descriptor)
    return Path(filename)


def _validate_capture(path: Path) -> Path:
    try:
        if not path.is_file() or path.stat().st_size < MINIMUM_CAPTURE_BYTES:
            raise RemoteCaptureError("el sistema no produjo una imagen válida")
        os.chmod(path, 0o600)
    except OSError as error:
        raise RemoteCaptureError("no se pudo preparar la imagen capturada") from error
    return path


def capture_screen(directory: Path) -> Path:
    """Capture the active macOS display without interpreting any user input."""

    if os.name == "nt" or not Path(SCREEN_CAPTURE_BINARY).is_file():
        raise RemoteCaptureError("la captura de pantalla no está disponible en este sistema")
    path = _new_capture_path(directory, "screen-")
    try:
        result = subprocess.run(
            [SCREEN_CAPTURE_BINARY, "-x", "-t", "png", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=CAPTURE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RemoteCaptureError("macOS no autorizó la captura de pantalla")
        return _validate_capture(path)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RemoteCaptureError("no se pudo capturar la pantalla") from error
    except Exception:
        discard_capture(path, directory)
        raise


def capture_camera(directory: Path) -> Path:
    """Capture one local webcam frame when the installed interpreter has TCC access."""

    try:
        import cv2  # Imported only when a locally approved camera request arrives.
    except ImportError as error:
        raise RemoteCaptureError("la cámara requiere OpenCV en el entorno de Telegram") from error
    path = _new_capture_path(directory, "camera-")
    camera = None
    try:
        backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
        camera = cv2.VideoCapture(0, backend)
        if not camera.isOpened():
            raise RemoteCaptureError("macOS no autorizó la cámara o no hay cámara disponible")
        frame = None
        deadline = time.monotonic() + CAMERA_WARMUP_SECONDS
        while time.monotonic() < deadline:
            available, candidate = camera.read()
            if available and _is_meaningfully_visible_frame(candidate):
                frame = candidate
            time.sleep(CAMERA_FRAME_INTERVAL_SECONDS)
        if frame is None:
            raise RemoteCaptureError(
                "la cámara no produjo una señal visible; revisa el permiso de Cámara de macOS o la fuente de vídeo"
            )
        if not cv2.imwrite(str(path), frame):
            raise RemoteCaptureError("la cámara no produjo una imagen válida")
        return _validate_capture(path)
    except RemoteCaptureError:
        discard_capture(path, directory)
        raise
    except Exception as error:
        discard_capture(path, directory)
        raise RemoteCaptureError("no se pudo capturar la cámara") from error
    finally:
        if camera is not None:
            camera.release()


def _is_meaningfully_visible_frame(frame: object) -> bool:
    """Reject all-black frames instead of leaking a false-success image to Telegram."""

    try:
        maximum = int(frame.max())
        mean = float(frame.mean())
    except (AttributeError, TypeError, ValueError):
        return False
    return maximum >= MINIMUM_VISIBLE_FRAME_MAX and mean >= MINIMUM_VISIBLE_FRAME_MEAN


def discard_capture(path: Path, directory: Path) -> None:
    """Delete only a capture created inside the dedicated private directory."""

    try:
        resolved_directory = directory.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
        if resolved_path.parent == resolved_directory and resolved_path.suffix == ".png":
            resolved_path.unlink(missing_ok=True)
    except OSError:
        return
