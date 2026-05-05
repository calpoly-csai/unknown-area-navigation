'''
Author: Ivan Torriani
Model: gpt-4o-mini (OpenAI API)
Description: This file uses GPT-4o-mini via the OpenAI API to convert
.jpg images to textual descriptions and traversability assessments in
JSON format. gpt-4o-mini is the cheapest OpenAI model with vision support.

Usage:
    python openaiDemo.py <path_to_image.jpg>

Requirements:
    pip install openai pillow python-dotenv
    OPENAI_API_KEY must be set in a .env file at the project root (or as
    an environment variable).
'''

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from helpers import parse_traversability

def _load_api_key() -> str:
    """
    Walk up the directory tree from this file looking for a .env that
    contains OPENAI_API_KEY, then return the key value.
    """
    for directory in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        env_file = directory / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            break

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY not found. "
            "Add it to your .env file or set it as an environment variable."
        )
        sys.exit(1)
    return api_key


def encode_image(image_path: str) -> str:
    """Base64-encode the image for the OpenAI vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    # ------------------------------------------------------------------ #
    # CLI argument parsing & input validation                             #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description="Run GPT-4o-mini scene analysis on a JPG image."
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

    # Validate image is openable
    try:
        Image.open(image_path).verify()
    except (IOError, Exception) as e:
        print(f"Error: Could not open image '{image_path}': {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # OpenAI client setup                                                  #
    # ------------------------------------------------------------------ #
    api_key = _load_api_key()
    client = OpenAI(api_key=api_key, timeout=30.0)
    image_data = encode_image(image_path)
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
    }

    # ------------------------------------------------------------------ #
    # Single API call: description, traversability, and justification     #
    # ------------------------------------------------------------------ #
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {
                            "type": "text",
                            "text": (
                                "Analyse this image for an autonomous vehicle. "
                                "Reply with a JSON object with exactly these fields:\n"
                                "- \"description\": one concise sentence describing the scene\n"
                                "- \"traversable\": true or false — is the scene safe and passable?\n"
                                "- \"justification\": one sentence explaining the traversability decision\n"
                                "Return only the raw JSON, no markdown."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        description = parsed["description"]
        traversability = bool(parsed["traversable"])
        justification = parsed["justification"]
    except Exception as e:
        print(f"Error: OpenAI API call failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # JSON output                                                          #
    # ------------------------------------------------------------------ #
    output = {
        "imageName": os.path.basename(image_path),
        "imageDescription": description,
        "traversability": traversability,
        "justification": justification,
    }

    output_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), "json_outputs")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_output.json")

    try:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
    except Exception as e:
        print(f"Error: Failed to write output file: {e}")
        sys.exit(1)

    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
