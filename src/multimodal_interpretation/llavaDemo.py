'''
Author: Ivan Torriani
Model: llava-hf/llava-1.5-7b-hf
Description: This file uses the LLaVA-1.5-7B vision-language model
to convert .jpg images to textual descriptions and traversability
assessments in JSON format.
'''

import argparse
import os
import sys
import json
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from helpers import parse_traversability, build_output, get_output_path


def main():
    # ------------------------------------------------------------------ #
    # CLI argument parsing & input validation                             #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description="Run LLaVA-1.5-7B scene analysis on a JPG image."
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
        image = Image.open(image_path)
    except (IOError, Image.UnidentifiedImageError) as e:
        print(f"Error: Could not open image '{image_path}': {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #
    try:
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Error: Failed to load LLaVA model: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Scene description generation                                         #
    # ------------------------------------------------------------------ #
    try:
        prompt_text = "USER: <image>\nDescribe the scene in this image in detail.\nASSISTANT:"
        inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=256)
        # Decode only the newly generated tokens (skip the input prompt)
        description = processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
    except Exception as e:
        print(f"Error: Model inference failed during description generation: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Traversability assessment                                            #
    # ------------------------------------------------------------------ #
    try:
        traversability_prompt = (
            "USER: <image>\nBased on this image, is the scene safe and passable for an "
            "autonomous vehicle? Answer with only 'yes' or 'no'.\nASSISTANT:"
        )
        inputs = processor(text=traversability_prompt, images=image, return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=16)
        # Decode only the newly generated tokens (skip the input prompt)
        traversability_response = processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
    except Exception as e:
        print(f"Error: Model inference failed during traversability assessment: {e}")
        sys.exit(1)

    traversability = parse_traversability(traversability_response)

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
