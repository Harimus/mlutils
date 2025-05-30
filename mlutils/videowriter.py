import logging

logger = logging.getLogger(__name__)
# This module provides a VideoWriter class for saving animations to video files.
# It uses OpenCV for video writing, and falls back to a dummy implementation if OpenCV is not available.
try:
    import cv2

    class VideoWriter:
        def __init__(self, output_file, frame_size, fps=30, codec="H264"):
            self.output_file = output_file + ".mp4"
            self.frame_size = frame_size
            self.fps = fps
            self.codec = cv2.VideoWriter.fourcc(*codec)
            self.out = None

        def __enter__(self):
            self.out = cv2.VideoWriter(
                self.output_file, self.codec, self.fps, self.frame_size
            )
            return self

        def write(self, frame):
            if self.out is None:
                raise RuntimeError(
                    "Video writer is not initialized. Use 'with save_animation(...) as sa:'"
                )

            # Ensure the frame size matches the expected size
            if (
                frame.shape[:2] != self.frame_size[::-1]
            ):  # frame_size is (w, h), but shape is (h, w)
                raise ValueError(
                    f"Frame size {frame.shape[:2]} does not match expected {self.frame_size[::-1]}."
                )
            # Convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.out.write(frame)

        def __exit__(self, *args):
            if self.out:
                self.out.release()

except ImportError:
    logger.warning("OpenCV not available. Using dummy VideoWriter implementation.")

    class VideoWriter:
        def __init__(self, output_file, frame_size, fps=30, codec="H264"):
            self.output_file = output_file + ".mp4"
            self.frame_size = frame_size
            self.fps = fps
            self.codec = codec
            logger.warning(
                "Dummy VideoWriter: OpenCV is not available. No video will be saved."
            )

        def __enter__(self):
            return self

        def write(self, frame):
            logger.warning(
                "Dummy VideoWriter: Frame not written. OpenCV is not available."
            )

        def __exit__(self, *args):
            pass
