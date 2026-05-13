'''
Author: Ivan Torriani
Model: gpt-4.1-nano (OpenAI API)
Description: This file uses GPT-4.1-nano via the OpenAI API to analyse all
test images (test1–test10) in the test_images folder and outputs one JSON
file per image containing:
  - imageName
  - imageDescription
  - traversability
  - justification
  - timeTaken  (seconds for the API call)

A summary JSON is also written with totalTime for the full batch.

gpt-4.1-nano is currently OpenAI's cheapest vision-capable model at
$0.10/1M input tokens and $0.40/1M output tokens — cheaper than gpt-4o-mini.

Usage:
    python openaiDemo.py

Requirements:
    pip install openai pillow python-dotenv
    OPENAI_API_KEY must be set in a .env file at the project root (or as
    an environment variable).
'''

import base64
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from helpers import build_output


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #
SCRIPT_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"
OUTPUT_DIR = TEST_IMAGES_DIR / "json_outputs"
IMAGE_NAMES = [f"test{i}" for i in range(1, 11)]
EXTENSIONS = [".jpg", ".jpeg"]


def _load_api_key() -> str:
    """Walk up the directory tree looking for a .env with OPENAI_API_KEY."""
    for directory in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
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


def find_image(name: str) -> Path | None:
    """Return the path for a test image, trying each supported extension."""
    for ext in EXTENSIONS:
        candidate = TEST_IMAGES_DIR / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def encode_image(image_path: Path) -> str:
    """Base64-encode the image for the OpenAI vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyse_image(client: OpenAI, image_path: Path) -> dict:
    """
    Run a single GPT-4o-mini call for the given image.
    Returns the output dict (imageName, imageDescription, traversability,
    justification, timeTaken).
    """
    # Validate image is openable
    try:
        Image.open(image_path).verify()
    except Exception as e:
        raise RuntimeError(f"Could not open image '{image_path}': {e}")

    image_data = encode_image(image_path)
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
    }

    start = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
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
    elapsed = time.perf_counter() - start

    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)

    return build_output(
        image_path=str(image_path),
        description=parsed["description"],
        traversability=bool(parsed["traversable"]),
        justification=parsed["justification"],
        time_taken=elapsed,
    )


def main():
    api_key = _load_api_key()
    client = OpenAI(api_key=api_key, timeout=30.0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_start = time.perf_counter()
    results = []

    for name in IMAGE_NAMES:
        image_path = find_image(name)
        if image_path is None:
            print(f"  Skipping {name}: no image found in {TEST_IMAGES_DIR}")
            continue

        print(f"  Processing {image_path.name} ...", end=" ", flush=True)
        try:
            output = analyse_image(client, image_path)
        except Exception as e:
            print(f"FAILED — {e}")
            continue

        print(f"done ({output['timeTaken']:.2f}s)")
        results.append(output)

    total_time = round(time.perf_counter() - batch_start, 3)

    # Write consolidated output file
    out_path = OUTPUT_DIR / "openai_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nBatch complete: {len(results)} images in {total_time:.2f}s")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
