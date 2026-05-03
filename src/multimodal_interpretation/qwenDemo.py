'''
Author: Ivan Torriani
Description: This file uses the Qwen-VL vision-language model
to convert .jpg images to textual descriptions and traversability
assessments in JSON format.

Additional Notes: Extremely slow
'''

import argparse
import os
import sys
import json
from PIL import Image
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration


def parse_traversability(response: str) -> bool:
    """
    Parse the model's traversability response into a boolean.

    Returns True if the response starts with 'yes' (case-insensitive),
    False for all other responses including ambiguous ones (Requirement 3.3).
    """
    return response.strip().lower().startswith("yes")


def build_output(image_path: str, description: str, traversability: bool) -> dict:
    """
    Assemble the output dictionary for JSON serialisation.

    The returned dict contains exactly three fields:
      - imageName: basename of image_path (no directory component)
      - imageDescription: natural-language scene description
      - traversability: boolean safety assessment
    """
    return {
        "imageName": os.path.basename(image_path),
        "imageDescription": description,
        "traversability": traversability,
    }


def main():
    # ------------------------------------------------------------------ #
    # CLI argument parsing & input validation                             #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description="Run Qwen-VL scene analysis on a JPG image."
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
        processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    except Exception as e:
        print(f"Error: Failed to load Qwen-VL model: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Scene description generation                                         #
    # ------------------------------------------------------------------ #
    try:
        description_prompt = "Describe the scene in this image in detail."
        # Qwen2-VL requires the chat template format with an image content block
        # so the processor can correctly align image tokens with image features.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": description_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=256)
        # Decode only the newly generated tokens (skip the input prompt)
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        description = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": traversability_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=16)
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        traversability_response = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
    except Exception as e:
        print(f"Error: Model inference failed during traversability assessment: {e}")
        sys.exit(1)

    traversability = parse_traversability(traversability_response)

    # ------------------------------------------------------------------ #
    # JSON output                                                          #
    # ------------------------------------------------------------------ #
    output = build_output(image_path, description, traversability)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_name}_output.json"
    output_path = os.path.join(os.path.dirname(image_path), output_filename)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
