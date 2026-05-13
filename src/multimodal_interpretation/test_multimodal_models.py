'''
Author: Ivan Torriani
Description: Experiment runner that benchmarks two vision-language models
(gemini-2.5-flash-lite and gpt-4.1-nano) against the same set of test images
(test1–test10) for the methods section of the paper.

Each model analyses every image and produces:
  - imageName
  - imageDescription
  - traversability  (bool)
  - justification
  - timeTaken       (seconds for the API call)

Output files written to test_images/json_outputs/:
  - gemini_test.json
  - openai_test.json

Accuracy (traversability correctness + description quality) is assessed
manually by the researcher. timeTaken is recorded automatically.

Usage:
    python test_multimodal_models.py

Requirements:
    pip install google-generativeai openai pillow python-dotenv
    GEMINI_API_KEY and OPENAI_API_KEY must be set in a .env file at the
    project root (or as environment variables).
'''

import base64
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv

from helpers import build_output


# ------------------------------------------------------------------ #
# Paths & constants                                                    #
# ------------------------------------------------------------------ #
SCRIPT_DIR      = Path(__file__).resolve().parent
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"
OUTPUT_DIR      = TEST_IMAGES_DIR / "json_outputs"
IMAGE_NAMES     = [f"test{i}" for i in range(1, 11)]
EXTENSIONS      = [".jpg", ".jpeg"]

TRAVERSABILITY_PROMPT = (
    "Analyse this image for an autonomous vehicle. "
    "Reply with a JSON object with exactly these fields:\n"
    '- "description": one concise sentence describing the scene\n'
    '- "traversable": true or false — is the scene safe and passable?\n'
    '- "justification": one sentence explaining the traversability decision\n'
    "Return only the raw JSON, no markdown fences."
)


# ------------------------------------------------------------------ #
# Shared utilities                                                     #
# ------------------------------------------------------------------ #
def _load_env() -> None:
    """Walk up the directory tree and load the first .env found."""
    for directory in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        env_file = directory / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            break


def find_image(name: str) -> Path | None:
    """Return the path for a test image, trying each supported extension."""
    for ext in EXTENSIONS:
        candidate = TEST_IMAGES_DIR / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences a model may add despite instructions."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


# ------------------------------------------------------------------ #
# Gemini runner                                                        #
# ------------------------------------------------------------------ #
def _run_gemini(image_names: list[str]) -> list[dict]:
    """Run gemini-2.5-flash-lite over all images and return result dicts."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("  [Gemini] google-generativeai not installed — skipping.")
        return []

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY")
    if not api_key:
        print("  [Gemini] GEMINI_API_KEY not found — skipping.")
        return []

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    results = []

    for name in image_names:
        image_path = find_image(name)
        if image_path is None:
            print(f"  [Gemini] Skipping {name}: image not found")
            continue

        print(f"  [Gemini] {image_path.name} ...", end=" ", flush=True)
        try:
            img = Image.open(image_path).convert("RGB")
            start = time.perf_counter()
            response = model.generate_content([TRAVERSABILITY_PROMPT, img])
            elapsed = time.perf_counter() - start

            parsed = json.loads(_strip_fences(response.text))
            output = build_output(
                image_path=str(image_path),
                description=parsed["description"],
                traversability=bool(parsed["traversable"]),
                justification=parsed["justification"],
                time_taken=elapsed,
            )
            print(f"done ({output['timeTaken']:.2f}s)")
            results.append(output)
        except Exception as e:
            print(f"FAILED — {e}")

    return results


# ------------------------------------------------------------------ #
# OpenAI runner                                                        #
# ------------------------------------------------------------------ #
def _run_openai(image_names: list[str]) -> list[dict]:
    """Run gpt-4.1-nano over all images and return result dicts."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  [OpenAI] openai not installed — skipping.")
        return []

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  [OpenAI] OPENAI_API_KEY not found — skipping.")
        return []

    client = OpenAI(api_key=api_key, timeout=30.0)
    results = []

    for name in image_names:
        image_path = find_image(name)
        if image_path is None:
            print(f"  [OpenAI] Skipping {name}: image not found")
            continue

        print(f"  [OpenAI] {image_path.name} ...", end=" ", flush=True)
        try:
            # Validate image before encoding
            Image.open(image_path).verify()
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            start = time.perf_counter()
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                            {"type": "text", "text": TRAVERSABILITY_PROMPT},
                        ],
                    }
                ],
                max_tokens=200,
            )
            elapsed = time.perf_counter() - start

            parsed = json.loads(_strip_fences(response.choices[0].message.content))
            output = build_output(
                image_path=str(image_path),
                description=parsed["description"],
                traversability=bool(parsed["traversable"]),
                justification=parsed["justification"],
                time_taken=elapsed,
            )
            print(f"done ({output['timeTaken']:.2f}s)")
            results.append(output)
        except Exception as e:
            print(f"FAILED — {e}")

    return results


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #
def main() -> None:
    _load_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Gemini ──────────────────────────────────────────────────────
    print("\n=== Gemini (gemini-2.5-flash-lite) ===")
    gemini_start = time.perf_counter()
    gemini_results = _run_gemini(IMAGE_NAMES)
    gemini_time = round(time.perf_counter() - gemini_start, 3)

    gemini_out = OUTPUT_DIR / "gemini_test.json"
    with open(gemini_out, "w") as f:
        json.dump(gemini_results, f, indent=4)
    print(f"  → {len(gemini_results)} images in {gemini_time:.2f}s")
    print(f"  → Written to: {gemini_out}")

    # ── OpenAI ──────────────────────────────────────────────────────
    print("\n=== OpenAI (gpt-4.1-nano) ===")
    openai_start = time.perf_counter()
    openai_results = _run_openai(IMAGE_NAMES)
    openai_time = round(time.perf_counter() - openai_start, 3)

    openai_out = OUTPUT_DIR / "openai_test.json"
    with open(openai_out, "w") as f:
        json.dump(openai_results, f, indent=4)
    print(f"  → {len(openai_results)} images in {openai_time:.2f}s")
    print(f"  → Written to: {openai_out}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n=== Experiment complete ===")
    print(f"  Gemini : {len(gemini_results)} images, total {gemini_time:.2f}s")
    print(f"  OpenAI : {len(openai_results)} images, total {openai_time:.2f}s")


if __name__ == "__main__":
    main()
