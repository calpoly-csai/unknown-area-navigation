# Implementation Plan: qwen-multimodal-demo

## Overview

Implement `qwenDemo.py` in `src/multimodal_interpretation/` as a single-file Python script that uses the Qwen-VL vision-language model to process a JPG image and produce a structured JSON output. The implementation follows the same conventions as `moondream.py` in the same directory.

## Tasks

- [x] 1. Set up script skeleton and file header
  - Create `src/multimodal_interpretation/qwenDemo.py` with the standard project docstring (Author and Description fields per README format)
  - Add all necessary imports: `argparse`, `os`, `sys`, `json`, `PIL.Image`, `transformers`
  - Add a `main()` function stub and `if __name__ == "__main__": main()` entry point
  - Add inline comment blocks marking each major processing section (model loading, image loading, description generation, traversability assessment, JSON output)
  - _Requirements: 5.1, 5.2_

- [x] 2. Implement CLI argument parsing and input validation
  - [x] 2.1 Implement argument parser and file validation
    - Use `argparse` to accept a single positional argument for the `.jpg` file path
    - Validate that the file exists with `os.path.exists()`; print a descriptive error and call `sys.exit(1)` if not
    - Validate that the file has a `.jpg` extension (case-insensitive) using `os.path.splitext()`; print a descriptive error and call `sys.exit(1)` if not
    - _Requirements: 1.1, 1.2_

  - [ ]* 2.2 Write property test for input validation
    - **Property 4: Ambiguous traversability responses always default to `false`** (traversability parser is a pure function — test it in isolation)
    - **Validates: Requirements 3.3**
    - Use `hypothesis` to generate arbitrary strings and assert that any string not starting with `"yes"` (case-insensitive) returns `False`

- [x] 3. Implement image loading
  - [x] 3.1 Implement Pillow image loader
    - Use `PIL.Image.open()` to load the validated image path
    - Wrap in try/except for `IOError` and `PIL.UnidentifiedImageError`; print a descriptive error and call `sys.exit(1)` on failure
    - _Requirements: 1.3_

- [x] 4. Implement model loading
  - [x] 4.1 Load Qwen-VL model and processor from HuggingFace
    - Use `AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")` and `AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")`
    - Wrap in try/except; print a descriptive error and call `sys.exit(1)` on any loading failure
    - _Requirements: 2.1, 2.4_

- [x] 5. Implement scene description generation
  - [x] 5.1 Implement description inference
    - Construct the description prompt: `"Describe the scene in this image in detail."`
    - Pass the image and prompt through the processor and run `model.generate()` with `max_new_tokens=256`
    - Decode the output with `processor.decode(..., skip_special_tokens=True)` to obtain the `imageDescription` string
    - Wrap inference in try/except; print a descriptive error and call `sys.exit(1)` on failure
    - _Requirements: 2.2, 2.3, 2.4_

- [x] 6. Implement traversability assessment
  - [x] 6.1 Implement traversability prompt and boolean parser
    - Send the traversability prompt to the model: `"Based on this image, is the scene safe and passable for an autonomous vehicle? Answer with only 'yes' or 'no'."`
    - Strip and lowercase the model response; set `traversability = True` if it starts with `"yes"`, `False` otherwise (covers ambiguous responses)
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 6.2 Write property test for traversability parser
    - **Property 4: Ambiguous traversability responses always default to `false`**
    - **Validates: Requirements 3.3**
    - Extract the parsing logic into a standalone helper function `parse_traversability(response: str) -> bool`
    - Use `hypothesis` to generate arbitrary strings; assert that only strings starting with `"yes"` (case-insensitive) return `True`, all others return `False`

  - [ ]* 6.3 Write unit tests for traversability parser
    - Test `parse_traversability("yes")` → `True`
    - Test `parse_traversability("Yes, it is safe")` → `True`
    - Test `parse_traversability("no")` → `False`
    - Test `parse_traversability("maybe")` → `False`
    - Test `parse_traversability("")` → `False`
    - _Requirements: 3.2, 3.3_

- [x] 7. Implement JSON output
  - [x] 7.1 Assemble and write the output JSON file
    - Build the output dict with exactly three fields: `imageName` (basename only, exact casing), `imageDescription`, and `traversability` (Python bool)
    - Derive the output filename: `{stem}_output.json` in the same directory as the input image
    - Write using `json.dump()` with `indent=4`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 7.2 Write property tests for JSON output assembly
    - **Property 1: Output JSON always contains exactly the three required fields**
    - **Validates: Requirements 4.2**
    - **Property 2: `imageName` is always the basename of the input path**
    - **Validates: Requirements 4.3, 4.4**
    - **Property 3: `traversability` is always a boolean**
    - **Validates: Requirements 4.2**
    - **Property 5: Output file is always co-located with the input image**
    - **Validates: Requirements 4.1, 4.5**
    - Extract output assembly into a helper function `build_output(image_path: str, description: str, traversability: bool) -> dict` and test it with `hypothesis`

  - [ ]* 7.3 Write unit tests for JSON output
    - Test that the output dict has exactly the keys `imageName`, `imageDescription`, `traversability`
    - Test that `imageName` is the basename (not full path) for various input paths
    - Test that the output filename is `{stem}_output.json`
    - Test that the output file is written to the same directory as the input
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 8. Checkpoint — wire everything together in `main()`
  - Connect all components in `main()`: parse args → load image → load model → generate description → assess traversability → write JSON
  - Print a success message to stdout indicating the output file path
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2_

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- `parse_traversability()` and `build_output()` should be extracted as standalone helper functions to enable unit and property testing without running the full model pipeline
- Property tests use the `hypothesis` library; install with `pip install hypothesis` if not already present
- All correctness properties are defined in `design.md` — each property test task references the relevant property number
- The Qwen-VL model is large; model loading tests should be skipped in CI unless a GPU environment is available
