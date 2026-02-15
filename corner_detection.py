#!/usr/bin/env python3
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class BoardDetection:
    lines_rho_theta: List[Tuple[float, float]]
    corners: np.ndarray  # shape (4, 2) ordered tl, tr, br, bl


def _normalize_rho_theta(rho: float, theta: float) -> Tuple[float, float]:
    # Keep rho positive for more consistent sorting
    if rho < 0:
        rho = -rho
        theta = (theta + np.pi) % np.pi
    return rho, theta


def _line_from_rho_theta(rho: float, theta: float) -> Tuple[float, float, float]:
    # x cos(theta) + y sin(theta) = rho  =>  a x + b y + c = 0
    a = np.cos(theta)
    b = np.sin(theta)
    c = -rho
    return a, b, c


def _intersect_lines(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Tuple[float, float]:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return float("nan"), float("nan")
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return x, y


def _order_points(pts: np.ndarray) -> np.ndarray:
    # Order points as: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def detect_board_lines_and_corners(image: np.ndarray) -> BoardDetection:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Light blur to reduce sensor noise and speckle without smearing edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 150)
    if lines is None or len(lines) < 4:
        raise RuntimeError("Not enough lines found for board detection.")

    vertical = []
    horizontal = []
    for line in lines:
        rho, theta = line[0]
        rho, theta = _normalize_rho_theta(rho, theta)
        # Horizontal lines are near pi/2, vertical near 0
        if abs(theta - np.pi / 2) < np.pi / 6:
            horizontal.append((rho, theta))
        else:
            vertical.append((rho, theta))

    if len(horizontal) < 2 or len(vertical) < 2:
        raise RuntimeError("Insufficient horizontal/vertical lines found.")

    # Pick outermost lines by rho within each group
    horizontal = sorted(horizontal, key=lambda x: x[0])
    vertical = sorted(vertical, key=lambda x: x[0])
    top = horizontal[0]
    bottom = horizontal[-1]
    left = vertical[0]
    right = vertical[-1]

    lines_rho_theta = [top, bottom, left, right]

    top_l = _line_from_rho_theta(*top)
    bottom_l = _line_from_rho_theta(*bottom)
    left_l = _line_from_rho_theta(*left)
    right_l = _line_from_rho_theta(*right)

    tl = _intersect_lines(top_l, left_l)
    tr = _intersect_lines(top_l, right_l)
    br = _intersect_lines(bottom_l, right_l)
    bl = _intersect_lines(bottom_l, left_l)

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    corners = _order_points(corners)

    return BoardDetection(lines_rho_theta=lines_rho_theta, corners=corners)


def draw_lines_and_corners(image: np.ndarray, lines_rho_theta: List[Tuple[float, float]], corners: np.ndarray) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]

    # Draw lines in red
    for rho, theta in lines_rho_theta:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Extend the line for visualization
        pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
        pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
        cv2.line(out, pt1, pt2, (0, 0, 255), 2)

    # Draw corners in red
    for (x, y) in corners:
        cv2.circle(out, (int(x), int(y)), 8, (0, 0, 255), -1)

    return out


def main() -> None:
    input_path = "example.png"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "corners.png")

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {input_path}")

    detection = detect_board_lines_and_corners(image)
    annotated = draw_lines_and_corners(image, detection.lines_rho_theta, detection.corners)
    cv2.imwrite(output_path, annotated)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
