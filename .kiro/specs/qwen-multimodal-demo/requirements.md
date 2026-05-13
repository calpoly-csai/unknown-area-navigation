# Requirements Document

## Introduction

This feature adds `qwenDemo.py` to `src/multimodal_interpretation/`, a demo script that uses the Qwen-VL vision-language model (via HuggingFace Transformers) to process a JPG image and produce a structured JSON output. The output captures the image name, a natural-language description of the scene, and a traversability assessment indicating whether the scene is safe for an autonomous vehicle to navigate. The script follows the same pattern established by `moondream.py` in the same directory.

## Glossary

- **QwenDemo**: The Python script (`qwenDemo.py`) being created.
- **Qwen_Model**: The Qwen-VL vision-language model loaded via HuggingFace Transformers (e.g., `Qwen/Qwen2-VL-7B-Instruct` or `Qwen/Qwen-VL-Chat`).
- **Input_Image**: A `.jpg` file provided as input to QwenDemo.
- **Image_Description**: The natural-language textual description of the Input_Image produced by the Qwen_Model.
- **Traversability**: A boolean value (`true` or `false`) indicating whether the scene depicted in the Input_Image is safe and passable for an autonomous vehicle.
- **Output_JSON**: The structured JSON file written by QwenDemo containing `imageName`, `imageDescription`, and `traversability` fields.
- **Traversability_Prompt**: The prompt sent to the Qwen_Model asking it to assess whether the scene is safe for autonomous vehicle navigation.

---

## Requirements

### Requirement 1: Accept a JPG Image as Input

**User Story:** As a developer on the autonomous driving project, I want QwenDemo to accept a JPG image file path as input, so that I can run scene analysis on any captured image.

#### Acceptance Criteria

1. THE QwenDemo SHALL accept a file path to a `.jpg` image as a command-line argument.
2. IF the provided file path does not exist or is not a `.jpg` file, THEN THE QwenDemo SHALL print a descriptive error message and exit with a non-zero status code.
3. WHEN a valid `.jpg` file path is provided, THE QwenDemo SHALL load the image using Pillow for processing.

---

### Requirement 2: Generate a Scene Description via Qwen-VL

**User Story:** As a developer, I want QwenDemo to pass the image through the Qwen-VL model and receive a written description, so that the scene content is captured in natural language.

#### Acceptance Criteria

1. THE QwenDemo SHALL load the Qwen_Model and its associated tokenizer/processor from HuggingFace Transformers at startup.
2. WHEN the Input_Image is loaded, THE QwenDemo SHALL send the image along with a descriptive prompt to the Qwen_Model to generate an Image_Description.
3. THE Qwen_Model SHALL produce an Image_Description that describes the visual content of the scene in plain English.
4. IF the Qwen_Model fails to load or inference fails, THEN THE QwenDemo SHALL print a descriptive error message and exit with a non-zero status code.

---

### Requirement 3: Assess Scene Traversability

**User Story:** As a developer on the autonomous driving project, I want QwenDemo to determine whether the scene is safe for the autonomous vehicle to traverse, so that the output can be used in downstream navigation decisions.

#### Acceptance Criteria

1. WHEN the Image_Description has been generated, THE QwenDemo SHALL send a Traversability_Prompt to the Qwen_Model asking whether the depicted scene is safe and passable for an autonomous vehicle.
2. THE QwenDemo SHALL parse the Qwen_Model's response to the Traversability_Prompt and derive a boolean Traversability value (`true` if safe/passable, `false` otherwise).
3. IF the Qwen_Model's traversability response is ambiguous or cannot be parsed as a clear yes/no, THEN THE QwenDemo SHALL default Traversability to `false`.

---

### Requirement 4: Produce a Structured JSON Output File

**User Story:** As a developer, I want QwenDemo to write the results to a structured JSON file, so that downstream systems can consume the scene analysis programmatically.

#### Acceptance Criteria

1. WHEN processing is complete, THE QwenDemo SHALL write an Output_JSON file to the same directory as the Input_Image.
2. THE Output_JSON SHALL contain exactly three fields: `imageName` (string), `imageDescription` (string), and `traversability` (boolean).
3. THE `imageName` field SHALL contain only the filename (not the full path) of the Input_Image (e.g., `"scene_001.jpg"`).
4. THE `imageName` field SHALL be the exact filename of the Input_Image as provided, preserving the original casing and extension.
5. THE Output_JSON file SHALL be named after the Input_Image with a `_output.json` suffix (e.g., `scene_001_output.json`).
6. THE Output_JSON SHALL be formatted with 2-space or 4-space indentation for human readability.

---

### Requirement 5: File Header and Code Documentation

**User Story:** As a project maintainer, I want QwenDemo to include the standard file header and inline documentation, so that the codebase remains consistent and publication-ready.

#### Acceptance Criteria

1. THE QwenDemo SHALL include a docstring at the top of the file with `Author` and `Description` fields, following the format defined in the project README.
2. THE QwenDemo SHALL include inline comments explaining each major processing step (model loading, image loading, description generation, traversability assessment, JSON output).
