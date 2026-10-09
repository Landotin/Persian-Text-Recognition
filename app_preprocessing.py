"""The inference-time glyph preparation copied from notebook Block 6."""

import numpy as np
from PIL import Image, ImageOps

OUTPUT_SIZE = 224
CONTENT_SIZE = 192

def prepare_image(image, output_size=OUTPUT_SIZE, content_size=CONTENT_SIZE):
    image = ImageOps.exif_transpose(image)
    if 'A' in image.getbands() or image.mode == 'P':
        rgba = image.convert('RGBA')
        white = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba).convert('RGB')
    gray = image.convert('L')
    pixels = np.asarray(gray)
    border = np.concatenate([pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]])
    if float(np.median(border)) < 127.5:
        gray = ImageOps.invert(gray)
    scale = min(content_size / gray.width, content_size / gray.height)
    resized_size = (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
    gray = gray.resize(resized_size, Image.Resampling.BILINEAR)
    canvas = Image.new('L', (output_size, output_size), 255)
    canvas.paste(gray, ((output_size - gray.width) // 2, (output_size - gray.height) // 2))
    return canvas
