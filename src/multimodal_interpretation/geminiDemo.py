'''
Author: Ivan Torriani
Model: gemini-2.0-flash (Google Gemini API)
Description: This file uses the Gemini vision-language model via the
Google Generative AI API to convert .jpg images to textual descriptions
and traversability assessments in JSON format.
Reflection: Unlike the local HuggingFace models (LLaVA, Moondream, InternVL,
Qwen), this runs inference in the cloud so it is dramatically faster —
no local GPU required. Ideal for edge devices that can reach the internet.

Usage:
    python geminiDemo.py <path_to_image.jpg>

Requirements:
    pip install google-generativeai pillow python-dotenv
    GEMINI_API_KEY must be set in a .env file at the project root (or as an
    environment variable).
'''

import argparse
import os
import sys
import json
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

from helpers import parse_traversability, build_output, get_output_path


# Load GEMINI_API_KEY from the nearest .env file walking up from this script
def _load_api_key() -> str:
    """
    Walk up the directory tree from this file looking for a .env that
    contains GEMINI_API_KEY, then return the key value.
    Raises SystemExit if the key cannot be found.
    """
    search_dir = Path(__file__).resolve().parent
    for directory in [search_dir, *search_dir.parents]:
        env_file = directory / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            break

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY")
    if not api_key:
        print(
            "Error: GEMINI_API_KEY (or GEMINI_KEY) not found. "
            "Add it to your .env file or set it as an environment variable."
        )
        sys.exit(1)
    return api_key


def main():
    # ------------------------------------------------------------------ #
    # CLI argument parsing & input validation                             #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description="Run Gemini scene analysis on a JPG image."
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

    # ------------------------------------------------------------------ #
    # Image loading                                                        #
    # ------------------------------------------------------------------ #
    try:
        image = Image.open(image_path).convert("RGB")
    except (IOError, Image.UnidentifiedImageError) as e:
        print(f"Error: Could not open image '{image_path}': {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Gemini client setup                                                  #
    # ------------------------------------------------------------------ #
    api_key = _load_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # ------------------------------------------------------------------ #
    # Scene description generation                                         #
    # ------------------------------------------------------------------ #
    try:
        description_prompt = "Describe the scene in this image in detail."
        response = model.generate_content([description_prompt, image])
        description = response.text.strip()
    except Exception as e:
        print(f"Error: Gemini API call failed during description generation: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Traversability assessment                                            #
    # ------------------------------------------------------------------ #
    try:
        traversability_prompt = (
            "Based on this image, is the scene safe and passable for an autonomous "
            "vehicle? Answer with only 'yes' or 'no'."
        )
        response = model.generate_content([traversability_prompt, image])
        traversability_response = response.text.strip()
    except Exception as e:
        print(f"Error: Gemini API call failed during traversability assessment: {e}")
        sys.exit(1)

    traversability = parse_traversability(traversability_response)

    # ------------------------------------------------------------------ #
    # JSON output                                                          #
    # ------------------------------------------------------------------ #
    output = build_output(image_path, description, traversability)
    output["model"] = "gemini-2.0-flash"

    output_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), "json_outputs")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"gemini_{base_name}_output.json")

    try:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
    except Exception as e:
        print(f"Error: Failed to write output file: {e}")
        sys.exit(1)

    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
