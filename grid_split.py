#!/usr/bin/env python3
import os
from typing import List

import cv2
import numpy as np


def split_into_grid(image: np.ndarray, grid_size: int = 8) -> List[List[np.ndarray]]:
    h, w = image.shape[:2]
    if h != w:
        raise ValueError(f"Image must be square, got {w}x{h}")

    tile_size = h // grid_size
    tiles: List[List[np.ndarray]] = []
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            y0 = r * tile_size
            x0 = c * tile_size
            tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size].copy()
            row.append(tile)
        tiles.append(row)
    return tiles


def main() -> None:
    input_path = os.path.join("outputs", "warped.png")
    output_dir = os.path.join("outputs", "tiles")
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {input_path}")

    tiles = split_into_grid(image, grid_size=8)
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            out_path = os.path.join(output_dir, f"tile_r{r}_c{c}.png")
            cv2.imwrite(out_path, tile)
            print(f"{out_path} {tile.shape}")


if __name__ == "__main__":
    main()
