from pathlib import Path

from core.detection.detection_engine import DetectionEngine


def main() -> None:
    engine = DetectionEngine()

    image_path = Path("assets/test_image.jpg")

    if not image_path.is_file():
        print(f"Test image not found: {image_path}")
        print("Place a JPG image at this location and run again.")
        return

    result = engine.detect_image(
        image_path=str(image_path),
        confidence=0.25,
    )

    print("\nVisionForge AI - Detection Results")
    print("-" * 40)

    print("Image dimensions:",
          result["image_width"], "x", result["image_height"])

    print("Detected objects:", result["object_count"])
    print("Prediction time:", result["inference_time_ms"], "ms")

    for item in result["detections"]:
        print(
            f'{item["label"]}: '
            f'{item["confidence"]:.2%} confidence'
        )

    output_path = Path("outputs/annotated_test_image.jpg")

    success = __import__("cv2").imwrite(
        str(output_path),
        result["annotated_image"],
    )

    if not success:
        raise RuntimeError("Could not save the annotated image.")

    print("Annotated image saved to:", output_path)


if __name__ == "__main__":
    main()