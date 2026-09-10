# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom chessboard corner detector robust to glare on washed-out prints.

Segments dark squares as blobs, fits a regular grid model to their centroids
(RANSAC-style via row/column clustering), predicts all square centers from the
model, then derives inner corners as midpoints between diagonally adjacent
squares. Falls back to OpenCV detectors first; the custom path only runs when
they fail.
"""

from __future__ import annotations

import cv2
import numpy as np


def find_chessboard_custom(
    gray: np.ndarray, pattern: tuple[int, int]
) -> tuple[bool, np.ndarray | None]:
    """Find inner chessboard corners; returns (ok, corners Nx1x2 row-major).

    The corner order is canonicalized so that square cell (0, 0) — the one
    diagonally below-right of the first corner — is always the darker cell.
    For boards whose two square counts sum to an odd number this removes the
    180-degree ordering ambiguity, which otherwise flips the solved board
    pose between images and silently corrupts hand-eye calibration.
    """
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern,
        None,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        try:
            found, corners = cv2.findChessboardCornersSB(gray, pattern, None)
        except cv2.error:
            found = False
    if not found:
        found, corners = _grid_from_blobs(gray, pattern)
    if not found:
        return False, None
    return True, _canonicalize_corner_order(gray, corners, pattern)


def _canonicalize_corner_order(
    gray: np.ndarray, corners: np.ndarray, pattern: tuple[int, int]
) -> np.ndarray:
    """Flip the corner grid 180 degrees when cell (0, 0) is the lighter cell."""
    cols, rows = pattern
    pts = corners.reshape(rows, cols, 2)
    h, w = gray.shape

    def cell_center(j: int, i: int) -> tuple[int, int]:
        # Cell (j, i) sits diagonally before corner (j, i); approximate its
        # center from the four surrounding corners, clamped to the image.
        ids = [(j, i), (j, i + 1), (j + 1, i), (j + 1, i + 1)]
        ids = [(min(a, rows - 1), min(b, cols - 1)) for a, b in ids]
        xy = np.mean([pts[a, b] for a, b in ids], axis=0)
        return int(np.clip(xy[0], 0, w - 1)), int(np.clip(xy[1], 0, h - 1))

    def cell_mean(j: int, i: int, half: int = 3) -> float:
        x, y = cell_center(j, i)
        patch = gray[max(0, y - half) : y + half + 1, max(0, x - half) : x + half + 1]
        return float(patch.mean()) if patch.size else 255.0

    # Compare the two cells flanking the first corner along the row direction.
    if cell_mean(0, 0) > cell_mean(0, 1):
        corners = corners[::-1].copy()
    return corners


def _grid_from_blobs(
    gray: np.ndarray, pattern: tuple[int, int]
) -> tuple[bool, np.ndarray | None]:
    cols, rows = pattern
    n_sq_x, n_sq_y = cols + 1, rows + 1

    dark = (gray < 110).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cents = []
    sizes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 8 < w < 80 and 8 < h < 80 and 0.4 < w / h < 2.5:
            cents.append((x + w / 2.0, y + h / 2.0))
            sizes.append((w + h) / 2.0)
    if len(cents) < n_sq_x * n_sq_y * 0.4:
        return False, None
    cents = np.asarray(cents)
    pitch = float(np.median(sizes))  # checkerboard cells are adjacent: pitch == square side

    # Cluster centroid x/y into grid lines via pitch-quantized voting.
    col_idx = np.round(cents[:, 0] / pitch).astype(int)
    row_idx = np.round(cents[:, 1] / pitch).astype(int)
    col_map = {}
    row_map = {}
    for (cx, cy), ci, ri in zip(cents, col_idx, row_idx):
        col_map.setdefault(ci, []).append(cx)
        row_map.setdefault(ri, []).append(cy)
    # Keep the n_sq_x / n_sq_y strongest lines.
    col_lines = sorted(col_map, key=lambda k: -len(col_map[k]))[:n_sq_x]
    row_lines = sorted(row_map, key=lambda k: -len(row_map[k]))[:n_sq_y]
    if len(col_lines) < n_sq_x or len(row_lines) < n_sq_y:
        return False, None
    col_lines = sorted(col_lines)
    row_lines = sorted(row_lines)
    # Consecutive quantized indices must be adjacent (no gaps allowed).
    if np.any(np.diff(col_lines) != 1) or np.any(np.diff(row_lines) != 1):
        return False, None

    col_pos = np.array([np.mean(col_map[k]) for k in col_lines])
    row_pos = np.array([np.mean(row_map[k]) for k in row_lines])

    # Inner corner (i, j), i in [0, cols), j in [0, rows): the meeting point of
    # squares (i, j), (i+1, j), (i, j+1), (i+1, j+1) in square-grid coords.
    # Corner x = midpoint between column lines i and i+1 (in square coords the
    # corner sits between square columns), i.e. between col_pos[i] and
    # col_pos[i+1]; likewise for rows.
    corners = np.zeros((rows * cols, 1, 2), dtype=np.float32)
    for j in range(rows):
        for i in range(cols):
            x = 0.5 * (col_pos[i] + col_pos[i + 1])
            y = 0.5 * (row_pos[j] + row_pos[j + 1])
            corners[j * cols + i, 0] = (x, y)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return True, corners
