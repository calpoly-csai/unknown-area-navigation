"""
Shared helper functions for the multimodal interpretation demo scripts.

This module provides pure utility functions used by all three VLM demo scripts
(llavaDemo.py, moondreamDemo.py, internVLDemo.py) and the existing qwenDemo.py:

  - parse_traversability: converts a model's text response to a boolean
  - build_output: assembles the output dict for JSON serialisation
  - get_output_path: constructs the co-located output JSON file path
  - is_jpg: validates that a file path has a .jpg extension
"""

import os


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
        "imageDescription": description.strip(),
        "traversability": traversability,
    }


def get_output_path(image_path: str) -> str:
    """
    Construct the output JSON file path co-located with the input image.

    The output file is named '{imageBaseName}_output.json' and is placed
    in the same directory as the input image (Requirement 5.1).
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(os.path.dirname(image_path), f"{base_name}_output.json")


def is_jpg(path: str) -> bool:
    """
    Return True iff the file at *path* has a .jpg extension (case-insensitive).

    Used by each demo script to validate the CLI-supplied image path
    before attempting to open it (Requirement 1.3).
    """
    _, ext = os.path.splitext(path)
    return ext.lower() == ".jpg"
