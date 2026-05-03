# Implementation Plan: Multimodal Demo Alternatives

## Overview

Implement three standalone VLM demo scripts (`llavaDemo.py`, `moondreamDemo.py`, `internVLDemo.py`) in `src/multimodal_interpretation/`, each mirroring the structure of `qwenDemo.py`. Each script accepts a `.jpg` image path via CLI, loads an open-weights VLM, generates a scene description and traversability boolean, and writes a JSON output file. Property-based tests using `hypothesis` validate the five correctness properties defined in the design.

## Tasks

- [x] 1. Implement shared helper functions and property-based tests
  - [x] 1.1 Create `src/multimodal_interpretation/helpers.py` with `parse_traversability`, `build_output`, and output-path construction logic
    - Implement `parse_traversability(response: str) -> bool` — returns `True` iff `response.strip().lower().startswith("yes")`
    - Implement `build_output(image_path: str, description: str, traversability: bool) -> dict` — returns dict with keys `imageName`, `imageDescription`, `traversability`
    - Implement `get_output_path(image_path: str) -> str` — returns `os.path.join(os.path.dirname(image_path), f"{base_name}_output.json")`
    - _Requirements: 3.4, 4.2, 4.3, 5.1, 5.2_

  - [ ]* 1.2 Write property test for `parse_traversability` (Property 1)
    - **Property 1: parse_traversability is True iff response starts with "yes" (case-insensitive)**
    - Use `hypothesis` `@given` with strings prefixed by any case variant of `"yes"` → assert `True`; arbitrary strings not starting with `"yes"` → assert `False`
    - **Validates: Requirements 4.2, 4.3**

  - [ ]* 1.3 Write property test for description whitespace stripping (Property 2)
    - **Property 2: description whitespace is stripped**
    - Use `hypothesis` `@given` with arbitrary strings with random leading/trailing whitespace; verify `build_output(..., description.strip(), ...) ["imageDescription"] == description.strip()`
    - **Validates: Requirements 3.4**

  - [ ]* 1.4 Write property test for `build_output` field completeness (Property 3)
    - **Property 3: build_output always produces exactly three fields**
    - Use `hypothesis` `@given` with arbitrary `image_path` strings, `description` strings, and `traversability` booleans; assert `set(result.keys()) == {"imageName", "imageDescription", "traversability"}`
    - **Validates: Requirements 5.2**

  - [ ]* 1.5 Write property test for output path co-location (Property 4)
    - **Property 4: output path is co-located with input image**
    - Use `hypothesis` `@given` with arbitrary valid file path strings; assert `os.path.dirname(output_path) == os.path.dirname(image_path)` and filename matches `{base}_output.json`
    - **Validates: Requirements 5.1**

  - [ ]* 1.6 Write property test for non-.jpg extension rejection (Property 5)
    - **Property 5: non-.jpg extensions are always rejected**
    - Extract the extension-check logic into a testable helper `is_jpg(path: str) -> bool`; use `hypothesis` `@given` with arbitrary extensions that are not `.jpg` (case-insensitive); assert `is_jpg` returns `False`
    - **Validates: Requirements 1.3**

- [x] 2. Checkpoint — Ensure all property tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Implement `llavaDemo.py`
  - [x] 3.1 Create `src/multimodal_interpretation/llavaDemo.py` with module-level docstring, CLI validation, and image loading
    - Add module-level docstring with author, model ID `llava-hf/llava-1.5-7b-hf`, and brief description matching `qwenDemo.py` style
    - Implement `argparse`-based CLI accepting one positional `image_path` argument
    - Validate path existence, `.jpg` extension (case-insensitive), and `PIL.Image.open` success; `sys.exit(1)` with descriptive messages on failure
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.4, 6.1, 6.2, 6.3_

  - [x] 3.2 Add model loading for LLaVA-1.5-7B
    - Load `AutoProcessor` and `LlavaForConditionalGeneration` from `llava-hf/llava-1.5-7b-hf` with `torch_dtype=torch.float16`
    - Wrap in `try/except`; print `"Error: Failed to load LLaVA model: {e}"` and `sys.exit(1)` on failure
    - _Requirements: 2.1, 2.2, 2.3, 7.1, 7.3, 7.4_

  - [x] 3.3 Add scene description generation for LLaVA
    - Build prompt using `USER: <image>\nDescribe the scene in this image in detail.\nASSISTANT:` template
    - Call `processor(text=prompt_text, images=image, return_tensors="pt")` then `model.generate(..., max_new_tokens=256)`
    - Decode only newly generated tokens; strip whitespace from result
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 3.4 Add traversability assessment and JSON output for LLaVA
    - Build traversability prompt using the same `USER: <image>\n...\nASSISTANT:` template with `max_new_tokens=16`
    - Call `parse_traversability` on the decoded response
    - Call `build_output` and `get_output_path`; write JSON with `indent=4`; print output path; `sys.exit(1)` on write failure
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 3.5 Write unit tests for `llavaDemo.py`
    - Test: zero args → exit 1; non-existent path → exit 1; non-`.jpg` extension → exit 1; invalid image content → exit 1
    - Test: mock `from_pretrained` to raise → exit 1; mock `generate` (description) to raise → exit 1; mock `generate` (traversability) to raise → exit 1; mock `open` to raise `PermissionError` → exit 1
    - Test: full success path with all model calls mocked → verify JSON fields, `indent=4`, and stdout message
    - Test: verify exact prompt strings and `max_new_tokens` values (256 / 16) and `torch_dtype=torch.float16`
    - _Requirements: 1.1–1.4, 2.3, 3.1–3.4, 4.1–4.4, 5.1–5.5, 7.3_

- [x] 4. Implement `moondreamDemo.py`
  - [x] 4.1 Create `src/multimodal_interpretation/moondreamDemo.py` with module-level docstring, CLI validation, and image loading
    - Add module-level docstring with author, model ID `vikhyatk/moondream2` (revision `2025-01-09`), and brief description
    - Implement identical CLI validation and image-loading logic as `llavaDemo.py`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.4, 6.1, 6.2, 6.3_

  - [x] 4.2 Add model loading for Moondream2
    - Load `AutoTokenizer` and `AutoModelForCausalLM` from `vikhyatk/moondream2` with `revision="2025-01-09"`, `trust_remote_code=True`, `torch_dtype=torch.float16`
    - Wrap in `try/except`; print `"Error: Failed to load Moondream2 model: {e}"` and `sys.exit(1)` on failure
    - _Requirements: 2.1, 2.2, 2.3, 7.1, 7.3, 7.4_

  - [x] 4.3 Add scene description and traversability using Moondream2's `query` API
    - Call `model.query(image, "Describe the scene in this image in detail.")` for description; strip whitespace
    - Call `model.query(image, "Based on this image, is the scene safe and passable for an autonomous vehicle? Answer with only 'yes' or 'no'.")` for traversability
    - Note: Moondream2's `query` method does not expose `max_new_tokens` directly — use the method as documented; wrap both calls in `try/except` with appropriate error messages
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4_

  - [x] 4.4 Add JSON output for Moondream2
    - Call `parse_traversability`, `build_output`, and `get_output_path`; write JSON with `indent=4`; print output path; `sys.exit(1)` on write failure
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 4.5 Write unit tests for `moondreamDemo.py`
    - Mirror the unit test structure from task 3.5, adapted for Moondream2's `query`-based API
    - Test: mock `model.query` to raise for description → exit 1; mock `model.query` to raise for traversability → exit 1
    - Test: full success path with mocked `query` → verify JSON fields and stdout message
    - Test: verify `trust_remote_code=True`, `revision="2025-01-09"`, and `torch_dtype=torch.float16` passed to `from_pretrained`
    - _Requirements: 1.1–1.4, 2.3, 3.1–3.4, 4.1–4.4, 5.1–5.5, 7.3_

- [x] 5. Implement `internVLDemo.py`
  - [x] 5.1 Create `src/multimodal_interpretation/internVLDemo.py` with module-level docstring, CLI validation, and image loading
    - Add module-level docstring with author, model ID `OpenGVLab/InternVL2-2B`, and brief description
    - Implement identical CLI validation and image-loading logic as the other two scripts
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.4, 6.1, 6.2, 6.3_

  - [x] 5.2 Add model loading for InternVL2-2B
    - Load `AutoTokenizer` and `AutoModel` from `OpenGVLab/InternVL2-2B` with `trust_remote_code=True`, `torch_dtype=torch.float16`
    - Wrap in `try/except`; print `"Error: Failed to load InternVL2 model: {e}"` and `sys.exit(1)` on failure
    - _Requirements: 2.1, 2.2, 2.3, 7.1, 7.3, 7.4_

  - [x] 5.3 Add scene description generation for InternVL2
    - Preprocess image into `pixel_values` using the model's `build_transform` utility
    - Call `model.chat(tokenizer, pixel_values, "Describe the scene in this image in detail.", generation_config)` with `max_new_tokens=256`; strip whitespace from result
    - Wrap in `try/except`; print `"Error: Model inference failed during description generation: {e}"` and `sys.exit(1)` on failure
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 5.4 Add traversability assessment and JSON output for InternVL2
    - Call `model.chat(...)` with the traversability prompt and `max_new_tokens=16`
    - Call `parse_traversability`, `build_output`, and `get_output_path`; write JSON with `indent=4`; print output path; `sys.exit(1)` on write failure
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 5.5 Write unit tests for `internVLDemo.py`
    - Mirror the unit test structure from task 3.5, adapted for InternVL2's `chat`-based API
    - Test: mock `model.chat` to raise for description → exit 1; mock `model.chat` to raise for traversability → exit 1
    - Test: full success path with mocked `chat` → verify JSON fields and stdout message
    - Test: verify `trust_remote_code=True` and `torch_dtype=torch.float16` passed to `from_pretrained`
    - _Requirements: 1.1–1.4, 2.3, 3.1–3.4, 4.1–4.4, 5.1–5.5, 7.3_

- [x] 6. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- `helpers.py` is a shared internal module imported by all three demo scripts; it is not a public API
- Property tests (tasks 1.2–1.6) validate pure functions only — model inference is not exercised by PBT
- Unit tests use `unittest.mock` to patch `from_pretrained`, `generate`/`query`/`chat`, and `open`
- Each script must remain fully self-contained for standalone use; `helpers.py` is the only shared dependency
- All error messages are printed to `stdout` (matching `qwenDemo.py` convention)
