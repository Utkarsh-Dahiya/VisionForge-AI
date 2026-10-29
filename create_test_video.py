import cv2
import numpy as np
from pathlib import Path

# Output location
output_path = Path("assets/test_video.mp4")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Video settings
width = 640
height = 480
fps = 20
duration_seconds = 5
total_frames = fps * duration_seconds

# Initialize MP4 video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create MP4 video. Check your OpenCV video codec support."
    )

# Generate a moving scene
for frame_number in range(total_frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Light background
    frame[:] = (235, 235, 235)

    # Moving rectangle
    x = 30 + (frame_number * 8) % 500
    cv2.rectangle(
        frame,
        (x, 150),
        (x + 100, 250),
        (0, 255, 0),
        -1
    )

    # Frame information
    cv2.putText(
        frame,
        f"VisionForge AI | Frame {frame_number + 1}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    writer.write(frame)

writer.release()

# Verify the generated file
if not output_path.exists() or output_path.stat().st_size == 0:
    raise RuntimeError("Test video generation failed.")

print("SUCCESS: Test video created!")
print(f"Path: {output_path.resolve()}")
print(f"Size: {output_path.stat().st_size:,} bytes")
print(f"Duration: {duration_seconds} seconds")
print(f"Resolution: {width}x{height}")
print(f"Frames: {total_frames}")