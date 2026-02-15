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


def _rho_theta_from_points(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    # Convert a line through two points into (rho, theta) for x cosθ + y sinθ = rho
    d = p2 - p1
    # Normal vector to the line
    n = np.array([d[1], -d[0]], dtype=np.float32)
    norm = np.hypot(n[0], n[1])
    if norm < 1e-6:
        return 0.0, 0.0
    n /= norm
    theta = np.arctan2(n[1], n[0])
    if theta < 0:
        theta += np.pi
    rho = n[0] * p1[0] + n[1] * p1[1]
    if rho < 0:
        rho = -rho
        theta = (theta + np.pi) % np.pi
    return float(rho), float(theta)


def _detect_corners_contour(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = image.shape[0] * image.shape[1]
    best = None
    best_area = 0.0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area < 0.05 * img_area or area > 0.95 * img_area:
            continue
        if area > best_area:
            best = approx
            best_area = area

    if best is None:
        return None

    corners = best.reshape(4, 2).astype(np.float32)
    return _order_points(corners)


def _cluster_line_orientations(lines: List[Tuple[float, float]]) -> List[int]:
    # Cluster line orientations into two dominant directions using k-means on (cos 2θ, sin 2θ).
    thetas = np.array([theta for _, theta in lines], dtype=np.float32)
    data = np.column_stack((np.cos(2 * thetas), np.sin(2 * thetas))).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    _, labels, _ = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(-1).tolist()


def _pick_outer_lines(lines: List[Tuple[float, float]], min_sep: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if len(lines) < 2:
        raise RuntimeError("Not enough lines in cluster.")
    lines_sorted = sorted(lines, key=lambda x: x[0])
    n = len(lines_sorted)

    if n >= 10:
        low_idx = max(0, int(n * 0.1))
        high_idx = min(n - 1, int(n * 0.9) - 1)
        if high_idx <= low_idx:
            low_idx, high_idx = 0, n - 1
    else:
        low_idx, high_idx = 0, n - 1

    l1 = lines_sorted[low_idx]
    l2 = lines_sorted[high_idx]

    if abs(l1[0] - l2[0]) < min_sep:
        l1, l2 = lines_sorted[0], lines_sorted[-1]
    return l1, l2


def detect_board_lines_and_corners(image: np.ndarray) -> BoardDetection:
    contour_corners = _detect_corners_contour(image)
    if contour_corners is not None:
        tl, tr, br, bl = contour_corners
        lines_rho_theta = [
            _rho_theta_from_points(tl, tr),
            _rho_theta_from_points(tr, br),
            _rho_theta_from_points(br, bl),
            _rho_theta_from_points(bl, tl),
        ]
        return BoardDetection(lines_rho_theta=lines_rho_theta, corners=contour_corners)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Light blur to reduce sensor noise and speckle without smearing edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 120)
    if lines is None or len(lines) < 4:
        raise RuntimeError("Not enough lines found for board detection.")

    all_lines: List[Tuple[float, float]] = []
    for line in lines:
        rho, theta = line[0]
        rho, theta = _normalize_rho_theta(rho, theta)
        all_lines.append((rho, theta))

    labels = _cluster_line_orientations(all_lines)
    cluster_a = [l for l, lab in zip(all_lines, labels) if lab == 0]
    cluster_b = [l for l, lab in zip(all_lines, labels) if lab == 1]

    if len(cluster_a) < 2 or len(cluster_b) < 2:
        raise RuntimeError("Insufficient line clusters found.")

    min_sep = 0.3 * min(image.shape[0], image.shape[1])
    l1a, l2a = _pick_outer_lines(cluster_a, min_sep)
    l1b, l2b = _pick_outer_lines(cluster_b, min_sep)

    lines_rho_theta = [l1a, l2a, l1b, l2b]

    # Assign "top/bottom" and "left/right" based on rho ordering in each orientation
    a_low, a_high = sorted([l1a, l2a], key=lambda x: x[0])
    b_low, b_high = sorted([l1b, l2b], key=lambda x: x[0])

    # We don't know which cluster is horizontal vs vertical; both are fine for intersections
    top = a_low
    bottom = a_high
    left = b_low
    right = b_high

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
