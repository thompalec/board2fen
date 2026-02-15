#!/usr/bin/env python3
import os
from typing import Tuple

import cv2
import numpy as np

from corner_detection import detect_board_lines_and_corners


def warp_to_square(image: np.ndarray, corners: np.ndarray, size: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    dst = np.array(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, H, (size, size))
    return warped, H


def main() -> None:
    input_path = "example.png"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "warped.png")

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {input_path}")

    detection = detect_board_lines_and_corners(image)
    warped, H = warp_to_square(image, detection.corners, size=800)
    cv2.imwrite(output_path, warped)
    print("Homography matrix:")
    print(H)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
