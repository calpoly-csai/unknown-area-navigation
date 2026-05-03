# Requirements Document

## Introduction

This feature adds three alternative vision-language model (VLM) demo scripts to the `src/multimodal_interpretation/` directory. Each script follows the same pattern as the existing `qwenDemo.py`: it accepts a `.jpg` image path as a CLI argument, loads a free-to-use (open-weights) VLM, generates a natural-language scene description, produces a boolean traversability assessment, and writes a JSON output file. Model selection prioritises inference speed and accuracy for scene understanding and traversability assessment tasks relevant to autonomous navigation.

## Glossary

- **Demo_Script**: A standalone Python script named `{ModelName}Demo.py` that wraps a single VLM for scene analysis.
- **VLM**: Vision-Language Model — a neural network that accepts image and text inputs and produces text outputs.
- **Traversability**: A boolean indicating whether a scene is safe and passable for an autonomous vehicle (`true` = passable, `false` = not passable).
- **JSON_Output**: A file named `{imageBaseName}_output.json` written to the same directory as the input image, containing `imageName`, `imageDescription`, and `traversability` fields.
- **CLI**: Command-Line Interface — the mechanism by which the user supplies the image path to the script.
- **Open_Weights_Model**: A VLM whose weights are publicly available and usable without a paid API key.
- **Processor**: The HuggingFace tokenizer/image-processor paired with a given model.

---

## Requirements

### Requirement 1: CLI Input and Validation

**User Story:** As a developer, I want each demo script to accept a `.jpg` image path via the CLI, so that I can run scene analysis from the terminal without modifying source code.

#### Acceptance Criteria

1. THE Demo_Script SHALL accept exactly one positional CLI argument representing the path to a `.jpg` image file.
2. WHEN the provided path does not exist on the filesystem, THE Demo_Script SHALL print a descriptive error message and exit with a non-zero status code.
3. WHEN the provided file does not have a `.jpg` extension (case-insensitive), THE Demo_Script SHALL print a descriptive error message and exit with a non-zero status code.
4. WHEN the provided file cannot be opened as a valid image, THE Demo_Script SHALL print a descriptive error message and exit with a non-zero status code.

---

### Requirement 2: Open-Weights Model Selection

**User Story:** As a developer, I want each demo to use a different free-to-use open-weights VLM, so that I can compare models without incurring API costs.

#### Acceptance Criteria

1. THE Demo_Script SHALL load its VLM exclusively from publicly available open-weights checkpoints (e.g., via HuggingFace Hub) that require no paid API key.
2. THE Demo_Script SHALL use a VLM distinct from `Qwen2-VL-7B-Instruct` and distinct from the models used in the other two new demo scripts.
3. WHEN the model or processor fails to load, THE Demo_Script SHALL print a descriptive error message and exit with a non-zero status code.
4. THE Demo_Script SHALL document the chosen model's HuggingFace model ID in a module-level docstring.

---

### Requirement 3: Scene Description Generation

**User Story:** As a developer, I want each demo to produce a natural-language description of the input image, so that I can understand what the model perceives in the scene.

#### Acceptance Criteria

1. WHEN a valid image is loaded, THE Demo_Script SHALL generate a natural-language scene description using the loaded VLM with a maximum of 256 new tokens.
2. THE Demo_Script SHALL use the prompt `"Describe the scene in this image in detail."` (or the model's equivalent instruction format) for description generation.
3. WHEN model inference fails during description generation, THE Demo_Script SHALL print a descriptive error message and exit with a non-zero status code.
4. THE Demo_Script SHALL strip leading and trailing whitespace from the generated description before storing it.

---

### Requirement 4: Traversability Assessment

**User Story:** As a developer, I want each demo to assess whether the scene is traversable for an autonomous vehicle, so that I can use the output in downstream navigation logic.

#### Acceptance Criteria

1. WHEN a valid image is loaded, THE Demo_Script SHALL generate a traversability response using the prompt `"Based on this image, is the scene safe and passable for an autonomous vehicle? Answer with only 'yes' or 'no'."` (or the model's equivalent instruction format) with a maximum of 16 new tokens.
2. WHEN the traversability response begins with `"yes"` (case-insensitive), THE Demo_Script SHALL set the traversability field to `true`.
3. WHEN the traversability response does not begin with `"yes"` (case-insensitive), THE Demo_Script SHALL set the traversability field to `false`.
4. WHEN model inference fails during traversability assessment, THE Demo_Script SHALL print a descriptive error message and exit with a non-zero status code.

---

### Requirement 5: JSON Output

**User Story:** As a developer, I want each demo to write a structured JSON file, so that downstream systems can consume the results programmatically.

#### Acceptance Criteria

1. THE Demo_Script SHALL write a JSON file named `{imageBaseName}_output.json` to the same directory as the input image upon successful completion.
2. THE JSON_Output SHALL contain exactly three fields: `imageName` (string — basename of the input file), `imageDescription` (string — the generated scene description), and `traversability` (boolean).
3. THE JSON_Output SHALL be formatted with 4-space indentation.
4. WHEN the JSON file cannot be written (e.g., permission error), THE Demo_Script SHALL print a descriptive error message and exit with a non-zero status code.
5. THE Demo_Script SHALL print the path of the written output file to stdout upon success.

---

### Requirement 6: Naming Convention and File Placement

**User Story:** As a developer, I want each demo script to follow a consistent naming convention and be placed in the correct directory, so that the project structure remains predictable.

#### Acceptance Criteria

1. THE Demo_Script SHALL be named `{ModelName}Demo.py` where `{ModelName}` is a concise identifier for the underlying VLM (e.g., `LlavaDemo.py`, `MoondreamDemo.py`, `InternVLDemo.py`).
2. THE Demo_Script SHALL be located at `src/multimodal_interpretation/{ModelName}Demo.py`.
3. THE Demo_Script SHALL include a module-level docstring stating the author field, the model used, and a brief description matching the style of `qwenDemo.py`.

---

### Requirement 7: Speed and Accuracy Prioritisation

**User Story:** As a developer, I want the selected models to prioritise inference speed and scene-understanding accuracy, so that the demos are practical for real-time or near-real-time autonomous navigation use cases.

#### Acceptance Criteria

1. THE Demo_Script SHALL use a VLM with a parameter count at or below 8 billion parameters to ensure feasibility on consumer-grade hardware.
2. WHERE a smaller model variant exists that achieves comparable scene-understanding accuracy, THE Demo_Script SHALL prefer the smaller variant to reduce inference latency.
3. THE Demo_Script SHALL load the model using `torch_dtype=torch.float16` or `torch_dtype="auto"` where supported, to reduce memory footprint and improve inference speed.
4. THE Demo_Script SHALL not load unnecessary model components (e.g., training-only heads) that would increase memory usage without improving inference quality.
