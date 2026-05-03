'''
Author: Ivan Torriani
Model: OpenGVLab/InternVL2-2B
Description: This file uses the InternVL2-2B vision-language model
to convert .jpg images to textual descriptions and traversability
assessments in JSON format.
'''

import argparse
import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from helpers import parse_traversability, build_output, get_output_path


def build_transform(input_size=448):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform


def main():
    # ------------------------------------------------------------------ #
    # CLI argument parsing & input validation                             #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description="Run InternVL2-2B scene analysis on a JPG image."
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
        tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL2-2B", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL2-2B", trust_remote_code=True,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Error: Failed to load InternVL2 model: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Image preprocessing                                                  #
    # ------------------------------------------------------------------ #
    transform = build_transform()
    pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(torch.float16)

    # ------------------------------------------------------------------ #
    # Scene description generation                                         #
    # ------------------------------------------------------------------ #
    try:
        generation_config = dict(max_new_tokens=256, do_sample=False)
        description = model.chat(
            tokenizer,
            pixel_values,
            "Describe the scene in this image in detail.",
            generation_config
        ).strip()
    except Exception as e:
        print(f"Error: Model inference failed during description generation: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Traversability assessment                                            #
    # ------------------------------------------------------------------ #
    try:
        traversability_prompt = (
            "Based on this image, is the scene safe and passable for an autonomous "
            "vehicle? Answer with only 'yes' or 'no'."
        )
        traversability_response = model.chat(
            tokenizer,
            pixel_values,
            traversability_prompt,
            dict(max_new_tokens=16, do_sample=False)
        )
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
