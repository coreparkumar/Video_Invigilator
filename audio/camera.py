"""Audio camera compatibility layer: re-exports open_camera from capture."""
from audio.capture import open_camera, MicSource, list_input_devices

__all__ = ["open_camera", "MicSource", "list_input_devices"]