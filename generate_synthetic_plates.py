"""
Generate synthetic Indian license plate images with ground truth text.

Usage:
    python generate_synthetic_plates.py --output_dir ./synthetic_plates --count 10

Creates a CSV file with the same name as the output directory:
    synthetic_plates.csv

Ground truth format:
    image_filename,ground_truth
    plate_001.png,DL4CAF1234
    ...

Note: This generates synthetic images for TESTING ONLY. It does not create
realistic Indian plates. For real evaluation, supply labeled plate images.
"""

from __future__ import annotations

import argparse
import csv
import random
import string
from pathlib import Path

import cv2
import numpy as np


# Indian registration format patterns
STATE_CODES = [
    "DL", "MH", "UP", "HR", "RJ", "KA", "TN", "AP", "TS", "JK", "BR", "GA",
    "MP", "OD", "PB", "CT", "JH", "AS", "ML", "MN", "OR", "WB", "SK", "HP",
    "CH", "DN", "AN", "UK", "PY", "NL",
]

# Common authority codes
AUTHORITY_CODES = [
    "A", "AB", "ABC", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]


def create_plate_image(text: str, output_path: Path,
                        width: int = 600, height: int = 200,
                        noise: float = 0.01, blur: bool = False,
                        show_text: bool = True) -> np.ndarray:
    """Create a synthetic plate image with the given text.

    Args:
        text: The ground-truth plate text (e.g., "DL4CAF1234")
        output_path: Where to save the image
        width: Image width in pixels
        height: Image height in pixels
        noise: Amount of Gaussian noise to add (0.0 to 1.0)
        blur: If True, add Gaussian blur
        show_text: If True, render the text on the plate

    Returns:
        The generated image as a numpy BGR array
    """
    # Create a dark background (typical Indian plate is black/yellow)
    plate = np.zeros((height, width, 3), dtype=np.uint8)
    plate.fill(0)  # Black background

    # Add random yellow accent
    plate[:, width//3:2*width//3] = [255, 255, 0]  # Yellow stripe

    # Create text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2

    # Calculate text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Position text centered horizontally, upper portion
    text_x = (width - text_size[0]) // 2
    text_y = text_size[1] + 20  # Offset from top

    if show_text:
        # Draw text in white
        cv2.putText(
            plate, text, (text_x, text_y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    # Add a border
    border = 5
    cv2.rectangle(plate, (border, border), (width - border, height - border),
                  (0, 0, 0), 2)

    # Add noise
    if noise > 0:
        noise_mat = np.random.normal(0, 255 * noise, plate.shape).astype(np.int16)
        plate = np.clip(plate.astype(np.int16) + noise_mat, 0, 255).astype(np.uint8)

    # Add blur
    if blur:
        plate = cv2.GaussianBlur(plate, (5, 5), 0)

    # Save
    cv2.imwrite(str(output_path), plate)

    return plate


def generate_random_plate_text() -> str:
    """Generate a random valid Indian registration plate text."""
    state = random.choice(STATE_CODES)
    district_digits = str(random.randint(1, 99)).zfill(2)  # 01-99
    authority_lengths = [1, 2, 3]
    authority_length = random.choice(authority_lengths)
    authority = "".join(random.choices(string.ascii_uppercase, k=authority_length))
    seq_digits = str(random.randint(1, 9999)).zfill(4)  # 0001-9999
    return f"{state}{district_digits}{authority}{seq_digits}"


def generate_dataset(
    output_dir: Path,
    num_images: int,
    include_noise: bool = True,
    include_blur: bool = False,
    image_width: int = 600,
    image_height: int = 200,
) -> Path:
    """Generate synthetic plate images and a ground-truth CSV.

    Returns:
        Path to the generated CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ground_truth.csv"
    rows = []

    for i in range(num_images):
        # Generate random plate text
        text = generate_random_plate_text()

        # Create filename
        filename = f"plate_{i+1:04d}{random.choice(['.png', '.jpg', '.jpeg'])}"
        filepath = output_dir / filename

        # Generate image
        include_noise_flag = include_noise and random.random() > 0.5
        include_blur_flag = include_blur and random.random() > 0.5

        create_plate_image(
            text=text,
            output_path=filepath,
            width=image_width,
            height=image_height,
            noise=0.02 if include_noise_flag else 0.0,
            blur=include_blur_flag,
            show_text=True,
        )

        rows.append({
            "image_filename": filename,
            "ground_truth": text,
        })

    # Write CSV
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename", "ground_truth"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {num_images} synthetic plate images in {output_dir}")
    print(f"Ground truth saved to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Indian license plate images for OCR evaluation."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./synthetic_plates"),
        help="Directory to store generated images and CSV.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of synthetic images to generate.",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Add synthetic noise to images.",
    )
    parser.add_argument(
        "--blur",
        action="store_true",
        help="Add synthetic blur to images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=600,
        help="Image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=200,
        help="Image height in pixels.",
    )

    args = parser.parse_args()

    print(f"Generating {args.count} synthetic plate images...")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Noise: {'yes' if args.noise else 'no'}")
    print(f"Blur: {'yes' if args.blur else 'no'}")
    print()

    generate_dataset(
        output_dir=args.output_dir,
        num_images=args.count,
        include_noise=args.noise,
        include_blur=args.blur,
        image_width=args.width,
        image_height=args.height,
    )


if __name__ == "__main__":
    main()