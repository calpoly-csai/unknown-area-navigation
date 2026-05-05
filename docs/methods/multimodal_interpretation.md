# Methods — Multimodal Scene Interpretation

## Overview

Even when an object is flagged as unknown, an autonomous agent still needs to make a navigation decision. This subsystem explores whether large vision-language models (VLMs) can provide richer, contextual understanding of a scene — specifically: a natural-language description of what is present, and a binary traversability judgment (safe to pass / not safe to pass).

---

## Study Design

This is an **exploratory, comparative study**. Models were each implemented as a standalone inference script and run against the same input images. Performance was assessed qualitatively, with a focus on output quality and practical feasibility for edge deployment. No ground-truth labels or quantitative metrics were used in this phase.

The study covers two categories of approach:
- **Local VLMs** — models loaded and run on-device via Hugging Face `transformers`
- **Cloud API VLMs** — inference delegated to a remote API (no local GPU required)

---

## Task Definition

Each model was given a single `.jpg` image and asked to produce:

1. **Scene description** — a concise natural-language description of what is present
2. **Traversability assessment** — is the scene safe and passable for an autonomous vehicle?
3. **Justification** — a one-sentence explanation of the traversability decision

### Local VLM prompts (sequential)

The local models used two separate inference calls:

> "Describe the scene in this image in detail."

> "Based on this image, is the scene safe and passable for an autonomous vehicle? Answer with only 'yes' or 'no'."

The traversability response was parsed into a boolean: `True` if the response began with "yes" (case-insensitive), `False` otherwise.

### OpenAI API prompt (single call)

The OpenAI demo uses a single structured prompt to reduce latency and cost:

> "Analyse this image for an autonomous vehicle. Reply with a JSON object with exactly these fields:
> - `description`: one concise sentence describing the scene
> - `traversable`: true or false — is the scene safe and passable?
> - `justification`: one sentence explaining the traversability decision"

### Output Format

```json
{
    "imageName": "<filename>.jpg",
    "imageDescription": "<concise scene description>",
    "traversability": true | false,
    "justification": "<one-sentence explanation>"
}
```

> Note: local VLM demos do not yet include the `justification` field.

---

## Models Evaluated

### Local VLMs (Hugging Face)

All local models were loaded via the Hugging Face `transformers` library in `float16` precision to reduce memory footprint.

| Model | Identifier | Parameters | Prompt Format |
|---|---|---|---|
| **Qwen2-VL** | `Qwen/Qwen2-VL-7B-Instruct` | ~7B | Chat template with image + text content blocks |
| **LLaVA-1.5** | `llava-hf/llava-1.5-7b-hf` | ~7B | `USER: <image>\n...\nASSISTANT:` |
| **Moondream2** | `vikhyatk/moondream2` (rev. 2025-01-09) | ~1.9B | `.query(image, prompt)` API |
| **InternVL2-2B** | `OpenGVLab/InternVL2-2B` | ~2B | `.chat(tokenizer, pixel_values, prompt, config)` API |

### Cloud API VLMs

| Model | Provider | Script | Notes |
|---|---|---|---|
| **GPT-4o-mini** | OpenAI | `openaiDemo.py` | Single structured API call; returns description, traversability, and justification in one response |
| **Gemini 2.0 Flash Lite** | Google | `geminiDemo.py` | Two sequential API calls for description and traversability |

---

## Inference Procedure

### Local VLMs

For each model and each input image:

1. The image was loaded with Pillow (`PIL.Image.open`).
2. Any model-specific preprocessing was applied:
   - **InternVL2-2B:** resized to 448×448 with BICUBIC interpolation, converted to tensor, normalized with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
   - **Qwen2-VL:** image passed as a content block within the chat template.
   - **LLaVA / Moondream2:** image passed directly to the processor or `.query()` API.
3. The scene description prompt was passed to the model; up to **256 new tokens** were generated.
4. The traversability prompt was passed to the model; up to **16 new tokens** were generated.
5. Only newly generated tokens (beyond the input prompt length) were decoded to avoid echoing the prompt in the output.
6. The traversability string was parsed to a boolean via `parse_traversability()`.
7. Results were written to a JSON file co-located with the input image.

### OpenAI API (GPT-4o-mini)

1. The image was base64-encoded and passed inline as a `data:image/jpeg;base64,...` URL.
2. A single chat completion request was made with a structured prompt requesting a JSON response containing `description`, `traversable`, and `justification`.
3. The response was parsed directly as JSON — no separate traversability parsing step needed.
4. Results were written to `json_outputs/<imageName>_output.json` relative to the input image directory.

### Gemini API (Gemini 2.0 Flash Lite)

1. The image was loaded with Pillow and passed directly to the `generate_content` call.
2. Two sequential API calls were made — one for description, one for traversability.
3. The traversability string was parsed to a boolean via `parse_traversability()`.
4. Results were written to `json_outputs/<imageName>_output.json` relative to the input image directory.

---

## Shared Utilities

A shared helper module (`src/multimodal_interpretation/helpers.py`) was developed to ensure consistent behavior across all four VLM implementations:

| Function | Purpose |
|---|---|
| `parse_traversability(response)` | Parses model text output to a boolean |
| `build_output(image_path, description, traversability)` | Assembles the output dict |
| `get_output_path(image_path)` | Constructs the co-located output JSON path |
| `is_jpg(path)` | Validates `.jpg` file extension before loading |

---

## Correctness Validation (Property-Based Testing)

The shared utilities were validated using **property-based testing** with the [Hypothesis](https://hypothesis.readthedocs.io/) library. Five correctness properties were defined and each tested against 100 randomly generated examples:

| Property | Description |
|---|---|
| P1 | `parse_traversability` returns `True` iff the response starts with "yes" (any case) |
| P2 | `build_output` strips leading/trailing whitespace from the description field |
| P3 | `build_output` always produces exactly the keys `{imageName, imageDescription, traversability}` |
| P4 | `get_output_path` returns a path in the same directory as the input, named `{base}_output.json` |
| P5 | `is_jpg` returns `False` for any extension that is not `.jpg` (case-insensitive) |

Tests are located in `src/multimodal_interpretation/test_helpers.py` and run via `pytest`.

---

## Study Materials

| Component | Details |
|---|---|
| Input images | `.jpg` files; two test images (`test1.jpg`, `test2.jpg`) in `src/multimodal_interpretation/test_images/` |
| Hardware (local) | CPU inference (no GPU acceleration; testing edge-device feasibility) |
| Hardware (API) | Cloud inference via OpenAI / Google APIs |
| Frameworks | Hugging Face `transformers`, PyTorch, torchvision, Pillow, `openai`, `google-generativeai` |
| Output | JSON per image, written to `json_outputs/` folder relative to the input image |

---

## Performance Observations

All four local VLMs exhibited inference times on the order of **10–30+ minutes per image** when run on CPU. This was observed consistently across all implementations and represents a critical feasibility constraint for real-time or near-real-time deployment on autonomous vehicle hardware.

By contrast, the cloud API approaches (GPT-4o-mini, Gemini 2.0 Flash Lite) return results in **2–5 seconds per image**, with no local GPU or model loading required. This makes them significantly more practical for prototyping and testing, though they introduce a dependency on network connectivity and third-party API availability.

These findings motivate future work on:

- Model quantization (e.g., INT8, GGUF) for faster local inference
- Hardware acceleration (GPU, NPU, or dedicated inference chips)
- Lighter-weight VLM alternatives designed for edge deployment

---

## Limitations

- No ground-truth traversability labels; assessment quality was evaluated qualitatively only.
- CPU-only inference; results may differ substantially with GPU acceleration.
- Traversability is assessed on individual static frames with no temporal context, depth information, or sensor fusion.
- The two test images used are researcher-supplied and do not constitute a representative benchmark.
