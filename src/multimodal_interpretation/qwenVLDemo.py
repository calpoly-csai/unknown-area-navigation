'''
Author: Ivan Torriani
Model: Qwen/Qwen2.5-VL-7B-Instruct (HuggingFace via Together AI provider)
Description: This file uses the Qwen2.5-VL vision-language model via
HuggingFace's router with the Together AI provider to convert .jpg images
to textual descriptions and traversability assessments in JSON format.

NOTE: This requires a HuggingFace Pro subscription or credits, as vision
models are not available on the free tier. The Together AI provider is
used for inference.

Usage:
    python qwenVLDemo.py <path_to_image.jpg>

Requirements:
    pip install openai pillow python-dotenv
    HUGGINGFACE_API_KEY must be set in a .env file at the project root
    (or as an environment variable). Requires HuggingFace Pro or credits.

Reflection: Uses OpenAI-compatible API format with a modern VL model
through HuggingFace's inference router with Together AI provider.
'''

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

from helpers import build_output, get_output_path

# Keywords that suggest the scene is NOT safely traversable.
_OBSTACLE_KEYWORDS = [
    "person", "people", "pedestrian", "crowd",
    "car", "vehicle", "truck", "bus", "motorcycle", "bicycle",
    "wall", "fence", "barrier", "gate",
    "water", "flood", "river", "puddle",
    "stairs", "steps", "cliff", "slope",
    "construction", "debris", "rubble",
    "dog", "cat", "animal",
]


def _load_api_key() -> str:
    """
    Walk up the directory tree from this file looking for a .env that
    contains HUGGINGFACE_API_KEY, then return the key value.
    """
    for directory in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        env_file = directory / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            break

    api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        print(
            "Error: HUGGINGFACE_API_KEY not found. "
            "Add it to your .env file or set it as an environment variable."
        )
        sys.exit(1)
    return api_key


def _resize_image_if_needed(image: Image.Image, max_size: int = 512) -> Image.Image:
    """
    Resize image if it's larger than max_size while maintaining aspect ratio.
    This helps avoid 413 Payload Too Large errors from the API.
    """
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image


def _encode_image(image_path: str) -> str:
    """
    Load, resize if needed, and base64-encode the image for the API.
    """
    image = Image.open(image_path).convert("RGB")
    image = _resize_image_if_needed(image)
    
    # Convert to bytes
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode("utf-8")


def _caption_to_traversability(caption: str) -> bool:
    """
    Derive a traversability boolean from a plain caption string.

    Returns False if any obstacle keyword is found in the caption,
    True otherwise. This is a best-effort heuristic because the model
    cannot answer direct traversability questions.
    """
    lower = caption.lower()
    return not any(kw in lower for kw in _OBSTACLE_KEYWORDS)


def main():
    # ------------------------------------------------------------------ #
    # CLI argument parsing & input validation                             #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description="Run Qwen2-VL scene captioning on a JPG image."
    )
    parser.add_argument("image_path", help="Path to the input .jpg image file.")
    args = parser.parse_args()

    image_path = args.image_path

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    _, ext = os.path.splitext(image_path)
    if ext.lower() != ".jpg":
        print(f"Error: Expected a .jpg file, got '{ext}' for: {image_path}")
        sys.exit(1)

    try:
        image = Image.open(image_path).convert("RGB")
    except (IOError, Image.UnidentifiedImageError) as e:
        print(f"Error: Could not open image '{image_path}': {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # HuggingFace OpenAI-compatible API setup                             #
    # ------------------------------------------------------------------ #
    api_key = _load_api_key()
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
        timeout=60.0
    )
    
    image_data = _encode_image(image_path)

    # ------------------------------------------------------------------ #
    # Caption generation via OpenAI-compatible API                        #
    # ------------------------------------------------------------------ #
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct:together",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one detailed sentence."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        
        description = response.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"Error: HuggingFace API call failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Traversability assessment (heuristic)                               #
    # ------------------------------------------------------------------ #
    traversability = _caption_to_traversability(description)

    # ------------------------------------------------------------------ #
    # JSON output                                                          #
    # ------------------------------------------------------------------ #
    output = build_output(image_path, description, traversability)
    output_path = get_output_path(image_path)

    try:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
    except Exception as e:
        print(f"Error: Failed to write output file: {e}")
        sys.exit(1)

    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
