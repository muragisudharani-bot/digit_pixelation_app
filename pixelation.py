from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def process_digit(raw_rgba_array: np.ndarray) -> np.ndarray:
    """Convert an RGBA/gray canvas image to a centered 32x32 uint8 grayscale array."""
    if raw_rgba_array is None:
        return np.zeros((32, 32), dtype=np.uint8)

    source = np.asarray(raw_rgba_array)
    if source.ndim == 2:
        gray = source.astype(np.uint8)
    elif source.ndim == 3:
        if source.shape[2] == 4:
            # Canvas RGBA: RGB luminance plus alpha. Alpha also makes a composited
            # white-on-black stroke usable while ignoring a transparent background.
            rgb = source[:, :, :3].astype(np.float32)
            alpha = source[:, :, 3:4].astype(np.float32) / 255.0
            gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) * alpha
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        elif source.shape[2] == 3:
            gray = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            return np.zeros((32, 32), dtype=np.uint8)
    else:
        return np.zeros((32, 32), dtype=np.uint8)

    # Step 2: threshold/bounding box based on actual nonzero ink.
    ys, xs = np.nonzero(gray)
    if xs.size == 0:
        return np.zeros((32, 32), dtype=np.uint8)

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    crop = gray[y_min:y_max + 1, x_min:x_max + 1]

    # Step 3: center the ink in a square, with 20% padding on the longest side.
    height, width = crop.shape
    side = int(np.ceil(max(height, width) * 1.20))
    side = max(side, 1)
    padded = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - height) // 2
    x_offset = (side - width) // 2
    padded[y_offset:y_offset + height, x_offset:x_offset + width] = crop

    # Step 4: area downsampling preserves proper intermediate grayscale values.
    resized = cv2.resize(padded, (32, 32), interpolation=cv2.INTER_AREA)
    return np.clip(resized, 0, 255).astype(np.uint8)


def render_pixel_grid_with_values(pixel_array: np.ndarray, upscale_to: int = 320) -> Image.Image:
    """Render a 32x32 pixel array with crisp blocks, grid lines, and intensity text."""
    pixels = np.asarray(pixel_array)
    if pixels.shape != (32, 32):
        raise ValueError("pixel_array must have shape (32, 32)")
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    # Overlay: nearest-neighbor scaling keeps every 32x32 block crisp.
    source = Image.fromarray(pixels, mode="L")
    scaled = source.resize((upscale_to, upscale_to), resample=Image.Resampling.NEAREST)
    grid = Image.merge("RGB", (scaled, scaled, scaled))
    draw = ImageDraw.Draw(grid)

    cell_size = upscale_to // 32

    # Thin grid lines between all 32x32 cells.
    for index in range(33):
        position = index * cell_size
        draw.line((position, 0, position, upscale_to), fill=(96, 96, 96), width=1)
        draw.line((0, position, upscale_to, position), fill=(96, 96, 96), width=1)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=7)
    except OSError:
        font = ImageFont.load_default()

    # Draw the original intensity in a contrasting color in each cell.
    for row in range(32):
        for col in range(32):
            value = int(pixels[row, col])
            text = str(value)
            text_color = (255, 255, 255) if value < 128 else (0, 0, 0)
            box = draw.textbbox((0, 0), text, font=font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]
            center_x = col * cell_size + cell_size // 2
            center_y = row * cell_size + cell_size // 2
            draw.text(
                (center_x - text_width // 2, center_y - text_height // 2),
                text,
                fill=text_color,
                font=font,
            )

    return grid
