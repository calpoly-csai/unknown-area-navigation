# Design Document: qwen-multimodal-demo

## Overview

`qwenDemo.py` is a Python script placed in `src/multimodal_interpretation/` that uses the Qwen-VL vision-language model (via HuggingFace Transformers) to process a JPG image and produce a structured JSON output. The script mirrors the pattern established by `moondream.py` in the same directory and is intended as a demo for the autonomous navigation project.

The script accepts a single JPG image path via the command line, loads the Qwen-VL model, generates a natural-language scene description, assesses traversability for an autonomous vehicle, and writes the results to a JSON file alongside the input image.

---

## Architecture

The script is a single-file, sequential Python program with no external modules beyond standard library and project dependencies. It follows a linear pipeline:

```
CLI argument → Image validation → Model loading → Description generation → Traversability assessment → JSON output
```

### Component Breakdown

| Component | Responsibility |
|---|---|
| Argument parser | Accept and validate the `.jpg` file path from CLI |
| Image loader | Load the image using Pillow |
| Model loader | Load Qwen-VL model and processor from HuggingFace |
| Description generator | Send image + prompt to model, receive scene description |
| Traversability assessor | Send follow-up prompt, parse boolean result |
| JSON writer | Assemble and write the output JSON file |

---

## Data Models

### Output JSON Schema

```json
{
  "imageName": "scene_001.jpg",
  "imageDescription": "A paved road with clear lane markings...",
  "traversability": true
}
```

| Field | Type | Description |
|---|---|---|
| `imageName` | string | Filename only (not full path), exact casing preserved |
| `imageDescription` | string | Natural-language description from Qwen-VL |
| `traversability` | boolean | `true` if scene is safe/passable, `false` otherwise |

---

## Implementation Details

### 1. File Header

The file begins with the standard project docstring:

```python
'''
Author: Ivan Torriani
Description: This file uses the Qwen-VL vision-language model
to convert .jpg images to textual descriptions and traversability
assessments in JSON format.
'''
```

### 2. Argument Parsing

Use Python's `argparse` module to accept a single positional argument: the path to the input `.jpg` file.

**Validation logic:**
- Check that the file exists using `os.path.exists()`
- Check that the file has a `.jpg` extension (case-insensitive) using `os.path.splitext()`
- On failure: print a descriptive error message and call `sys.exit(1)`

### 3. Image Loading

Use `PIL.Image.open()` to load the image. Wrap in a try/except to catch `IOError` or `UnidentifiedImageError`.

### 4. Model Loading

Load the Qwen-VL model using HuggingFace Transformers. The recommended model identifier is `Qwen/Qwen2-VL-7B-Instruct` (or `Qwen/Qwen-VL-Chat` as a fallback).

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
```

Wrap in a try/except to catch loading errors; print a descriptive message and `sys.exit(1)` on failure.

### 5. Description Generation

Construct a prompt asking the model to describe the scene, pass the image and prompt through the processor, run inference, and decode the output.

**Prompt (description):**
```
Describe the scene in this image in detail.
```

**Inference pattern:**
```python
inputs = processor(text=prompt, images=image, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=256)
description = processor.decode(output_ids[0], skip_special_tokens=True)
```

### 6. Traversability Assessment

Send a second prompt to the model asking about traversability.

**Traversability prompt:**
```
Based on this image, is the scene safe and passable for an autonomous vehicle? Answer with only 'yes' or 'no'.
```

**Parsing logic:**
- Strip and lowercase the model's response
- If it starts with `"yes"` → `traversability = True`
- If it starts with `"no"` → `traversability = False`
- Otherwise (ambiguous) → `traversability = False` (safe default per Requirement 3.3)

### 7. JSON Output

Assemble the output dictionary and write it using `json.dump()` with `indent=4`.

**Output file naming:**
```python
base_name = os.path.splitext(os.path.basename(image_path))[0]
output_filename = f"{base_name}_output.json"
output_path = os.path.join(os.path.dirname(image_path), output_filename)
```

---

## Error Handling Strategy

| Scenario | Behavior |
|---|---|
| File path does not exist | Print error, `sys.exit(1)` |
| File is not a `.jpg` | Print error, `sys.exit(1)` |
| Image cannot be opened by Pillow | Print error, `sys.exit(1)` |
| Model fails to load | Print error, `sys.exit(1)` |
| Model inference fails | Print error, `sys.exit(1)` |
| Traversability response is ambiguous | Default to `false` (no exit) |

---

## Correctness Properties

The following universal properties must hold for any valid execution of `qwenDemo.py`:

**Property 1: Output JSON always contains exactly the three required fields**
For any valid `.jpg` input, the output JSON file must contain exactly `imageName`, `imageDescription`, and `traversability` — no more, no fewer fields.

**Property 2: `imageName` is always the basename of the input path**
For any input path `p`, `output["imageName"] == os.path.basename(p)`. The full path is never stored.

**Property 3: `traversability` is always a boolean**
The `traversability` field in the output JSON is always a Python `bool` (`True` or `False`), never a string, integer, or `None`.

**Property 4: Ambiguous traversability responses always default to `false`**
For any model response string that does not start with `"yes"` (case-insensitive), the parsed traversability value is `False`.

**Property 5: Output file is always co-located with the input image**
For any input path `p`, the output JSON file is written to `os.path.dirname(p)` with the name `{stem}_output.json`.

---

## Dependencies

All dependencies are already present in `requirements.txt`:

| Package | Use |
|---|---|
| `pillow` | Image loading |
| `transformers` | Qwen-VL model and processor |
| `torch` | Model inference backend |

No new dependencies need to be added to `requirements.txt`.
