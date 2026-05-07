'''
Author: Ivan Torriani
Model: gemini-2.5-flash-lite (Google Gemini API)
Description: This file uses the Gemini vision-language model via the
Google Generative AI API to analyse all test images (test1–test10) in the
test_images folder and outputs one JSON file per image containing:
  - imageName
  - imageDescription
  - traversability
  - justification
  - timeTaken  (seconds for the API call)

A summary JSON is also written with totalTime for the full batch.

gemini-2.5-flash-lite is currently Google's cheapest and fastest model at
$0.10/1M input tokens and $0.40/1M output tokens. It also has a free tier
with up to 1,000 requests/day — ideal for research on a budget.

Usage:
    python geminiDemo.py

Requirements:
    pip install google-generativeai pillow python-dotenv
    GEMINI_API_KEY must be set in a .env file at the project root (or as an
    environment variable).
'''

import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

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
    """Walk up the directory tree looking for a .env with GEMINI_API_KEY."""
    for directory in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
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


def find_image(name: str) -> Path | None:
    """Return the path for a test image, trying each supported extension."""
    for ext in EXTENSIONS:
        candidate = TEST_IMAGES_DIR / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def analyse_image(model: genai.GenerativeModel, image_path: Path) -> dict:
    """
    Run a single Gemini call for the given image.
    Returns the output dict (imageName, imageDescription, traversability,
    justification, timeTaken).
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Could not open image '{image_path}': {e}")

    prompt = (
        "Analyse this image for an autonomous vehicle. "
        "Reply with a JSON object with exactly these fields:\n"
        "- \"description\": one concise sentence describing the scene\n"
        "- \"traversable\": true or false — is the scene safe and passable?\n"
        "- \"justification\": one sentence explaining the traversability decision\n"
        "Return only the raw JSON, no markdown fences."
    )

    start = time.perf_counter()
    response = model.generate_content([prompt, image])
    elapsed = time.perf_counter() - start

    raw = response.text.strip()
    # Strip markdown code fences if the model adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

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
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
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
            output = analyse_image(model, image_path)
        except Exception as e:
            print(f"FAILED — {e}")
            continue

        print(f"done ({output['timeTaken']:.2f}s)")
        results.append(output)

    total_time = round(time.perf_counter() - batch_start, 3)

    # Write consolidated output file
    out_path = OUTPUT_DIR / "gemini_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nBatch complete: {len(results)} images in {total_time:.2f}s")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
