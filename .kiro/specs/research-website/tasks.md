# Implementation Plan: Research Website

## Overview

Build a static, multi-page GitHub Pages site for the Unknown Area Navigation research project. The implementation proceeds in dependency order: shared stylesheet first, then the landing page, then each sub-page, and finally the property-based test suite. Every HTML file links to the single shared stylesheet and duplicates the navigation bar with the correct active link for that page.

---

## Tasks

- [x] 1. Create the shared CSS stylesheet (`assets/css/style.css`)
  - Create the `assets/css/` directory and `style.css` file
  - Add Google Fonts `<link>` import comment (the HTML files will include the `<link>` tag; the CSS declares the `font-family` stacks)
  - Define CSS custom properties (variables) for the full color palette: `--bg` `#0f1117`, `--surface` `#1a1d27`, `--surface-raised` `#22263a`, `--border` `#2e3350`, `--text-primary` `#e8eaf0`, `--text-secondary` `#8b92b0`, `--accent` `#5b8dee`, `--accent-hover` `#7aa3f5`, `--success-tint` `#1e3a2f`, `--error-tint` `#3a1e1e`
  - Implement CSS reset / base: `box-sizing: border-box`, `margin: 0`, `padding: 0`, `body` background and text color
  - Implement typography: `font-family` stack (Inter → system-ui → sans-serif), heading sizes and weights per the design table (`h1` 2.25rem/700, `h2` 1.6rem/600, `h3` 1.2rem/600), body `font-size: 1rem`, `line-height: 1.7`
  - Implement `.site-nav`: `position: sticky`, `top: 0`, surface background, border-bottom, flex layout, `z-index: 100`; `.site-nav a` styles (padding, color, font-size 0.9rem, font-weight 500, `border-bottom: 2px solid transparent`, transition); `.site-nav a:hover` and `.site-nav a.nav-active` (accent color + 2px bottom border)
  - Implement `.page-content`: `max-width: 860px`, `margin: 0 auto`, `padding: 2rem 1.5rem`
  - Implement `.site-footer`: surface background, border-top, centered text, secondary text color
  - Implement `.data-table` and `.table-wrapper`: `width: 100%`, `border-collapse: collapse`, `font-size: 0.9rem`; `th` with surface-raised background, borders, padding; `td` with borders, padding; `tbody tr:nth-child(even)` alternating shading; `tbody tr:hover`; `.result-correct` and `.result-incorrect` row classes
  - Implement `code` inline style: JetBrains Mono → Fira Code → Cascadia Code → monospace fallback, `font-size: 0.875em`, surface-raised background, accent-tinted color, padding, border-radius
  - Implement `.subsystem-grid` and `.subsystem-card`: CSS grid with `repeat(auto-fit, minmax(240px, 1fr))`, gap, card background, border, border-radius, padding; `.subsystem-card h3` in accent color
  - Implement `.table-wrapper`: `overflow-x: auto` for horizontal scroll on narrow viewports
  - Add `@media` query ensuring no horizontal overflow at any viewport width (no horizontal scrollbar at ≥ 1024px)
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 2. Build `index.html` — Landing Page
  - Create `index.html` at the repository root
  - Add the page shell: `<!DOCTYPE html>`, `<html lang="en">`, `<head>` with `<meta charset="UTF-8">`, `<meta name="viewport" content="width=device-width, initial-scale=1">`, `<title>Unknown Area Navigation</title>`, Google Fonts `<link>` for Inter and JetBrains Mono, `<link rel="stylesheet" href="assets/css/style.css">`
  - Add `<nav class="site-nav">` with five links using root-relative paths (`index.html`, `pages/unknown-object-detection.html`, `pages/multimodal-interpretation.html`, `pages/navigation.html`, `pages/experiments.html`); mark the Home link with `class="nav-active"`
  - Add `<main class="page-content">` with `<h1>Unknown Area Navigation</h1>`
  - Add `<section id="abstract"><h2>Abstract</h2>` containing the full abstract text from `docs/abstract.md`
  - Add `<section id="summary"><h2>Project Summary</h2>` with a ≤ 150-word paragraph stating (a) that deep learning models trained on fixed datasets fail to generalise to out-of-distribution environments, and (b) that this project develops a supplementary pipeline to address that limitation
  - Add `<section id="subsystems"><h2>Research Subsystems</h2>` with a `<div class="subsystem-grid">` containing three `.subsystem-card` divs — one per subsystem (Unknown Object Detection, Multimodal Scene Interpretation, AprilTag-Based Navigation) — each with an `<h3>`, a one-sentence description, and an `<a>` link to the corresponding sub-page
  - Add `<section id="links">` with an `<a href="https://github.com/ivantorriani/unknown-area-navigation">View on GitHub</a>` link
  - Add `<footer class="site-footer">` with project name and GitHub link
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 8.1, 8.6_

- [x] 3. Build `pages/unknown-object-detection.html`
  - Create `pages/unknown-object-detection.html`
  - Add the page shell with `<title>Unknown Object Detection — Unknown Area Navigation</title>`, Google Fonts links, and `<link rel="stylesheet" href="../assets/css/style.css">`
  - Add `<nav class="site-nav">` with five links using `../`-prefixed paths; mark the Unknown Object Detection link with `class="nav-active"`
  - Add `<section id="problem"><h2>Problem Statement</h2>` explicitly stating (a) that standard closed-vocabulary detectors assign the nearest known class label to out-of-distribution objects, and (b) that this silent mislabelling is the core problem the subsystem addresses
  - Add `<section id="approaches"><h2>Approaches</h2>` with four `<h3>` subsections:
    - Approach 1 — Confidence-Thresholded YOLO-World: one-sentence description, detection mechanism (flags predictions below 50% confidence as unknown), intended use case
    - Approach 2 — Dual-Model Consensus (NanoOWL + YOLO-World): one-sentence description, consensus mechanism (objects detected by NanoOWL but unmatched by YOLO-World flagged as unknown), intended use case
    - Approach 3 — Baseline Closed-Vocabulary YOLO: one-sentence description, mechanism (assigns nearest known class, no unknown flag), intended use case (illustrates the core limitation)
    - Approach 4 — Frame-to-Frame Colour Histogram Comparison: one-sentence description, flagging mechanism (Bhattacharyya distance between consecutive frames), intended use case
  - Add `<section id="parameters"><h2>Key Parameters</h2>` with a `<div class="table-wrapper"><table class="data-table">` containing columns Approach / Parameter Name / Value, populated with all 9 rows from the design's parameter table
  - Add `<section id="limitations"><h2>Limitations</h2>` with a `<ul>` explicitly covering: (a) qualitative evaluation with no quantitative benchmark, (b) NanoOWL run on CPU rather than NVIDIA Jetson, (c) planned custom class-agnostic model not completed
  - Add `<footer class="site-footer">`
  - _Requirements: 1.3, 1.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 8.1, 8.4, 8.5, 8.6_

- [x] 4. Build `pages/multimodal-interpretation.html`
  - Create `pages/multimodal-interpretation.html`
  - Add the page shell with `<title>Multimodal Scene Interpretation — Unknown Area Navigation</title>`, Google Fonts links, and `<link rel="stylesheet" href="../assets/css/style.css">`
  - Add `<nav class="site-nav">` with five links; mark the Multimodal Interpretation link with `class="nav-active"`
  - Add `<section id="task"><h2>Task Definition</h2>` describing the task: given a single image, produce a natural-language scene description and a binary traversability judgment (safe to pass / not safe to pass)
  - Add `<section id="models"><h2>Models Evaluated</h2>` with two subsections:
    - `<h3>Local VLMs</h3>` with a `<table class="data-table">` listing Qwen2-VL, LLaVA-1.5, Moondream2, InternVL2-2B with their parameter counts
    - `<h3>Cloud API VLMs</h3>` with a `<table class="data-table">` listing GPT-4o-mini (OpenAI) and Gemini 2.0 Flash Lite (Google) with their provider names
  - Add `<section id="performance"><h2>Performance Observations</h2>` stating that local VLMs required 10–30+ minutes per image on CPU while cloud API models returned results in 2–5 seconds
  - Add `<section id="utilities"><h2>Shared Utilities and Correctness Validation</h2>` describing the four helper functions (`parse_traversability`, `build_output`, `get_output_path`, `is_jpg`) using `<code>` elements for identifiers, and stating that each was validated with Hypothesis property-based testing against a minimum of 5 correctness properties, each tested against at least 100 randomly generated examples
  - Add `<section id="experiment-results"><h2>Comparative Experiment Results</h2>` summarising the Gemini vs. OpenAI accuracy (77.8% vs. 88.9%) and mean latency (1.944 s vs. 1.894 s) with a link to the Experiments page for full details
  - Add `<section id="limitations"><h2>Limitations</h2>` with a `<ul>` covering: no ground-truth labels, CPU-only inference, single static frames with no temporal context, non-representative two-image test set
  - Add `<footer class="site-footer">`
  - _Requirements: 1.3, 1.4, 3.1, 3.2, 3.3, 3.4, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 8.1, 8.4, 8.5, 8.6_

- [x] 5. Build `pages/navigation.html`
  - Create `pages/navigation.html`
  - Add the page shell with `<title>AprilTag-Based Navigation — Unknown Area Navigation</title>`, Google Fonts links, and `<link rel="stylesheet" href="../assets/css/style.css">`
  - Add `<nav class="site-nav">` with five links; mark the Navigation link with `class="nav-active"`
  - Add `<section id="task"><h2>Task Definition</h2>` describing: detect an AprilTag in a live camera feed, estimate its 3D pose using `cv2.solvePnP`, and issue one of five movement states — Move Left, Move Right, Move Forward, Stop, and No Tag Detected (emitted when no AprilTag is visible)
  - Add `<section id="calibration"><h2>Camera Calibration</h2>` covering: (a) checkerboard target dimensions (8×6 inner corners, 20 mm square size), (b) use of `cv2.findChessboardCorners` and `cv2.cornerSubPix` for corner detection, (c) use of `cv2.calibrateCamera` to compute camera matrix K and distortion coefficients, (d) coordinate convention (X right, Y down, Z into the scene); wrap all function names in `<code>` elements
  - Add `<section id="movement"><h2>Movement Control Logic</h2>` with a `<div class="table-wrapper"><table class="data-table">` with columns Condition / Evaluated In Order / Command, populated with all 5 rows from the design table (lateral < −0.075 m → Move Left; lateral > +0.075 m → Move Right; distance ≤ 0.1 m → Stop; otherwise → Move Forward; no tag → No Tag Detected); add a `<p class="table-note">` noting conditions are evaluated in the order listed
  - Add `<section id="limitations"><h2>Limitations</h2>` with a `<ul>` covering: single-tag assumption, discrete threshold-based commands with no PID controller, no temporal smoothing, no integration with the other two subsystems
  - Add `<footer class="site-footer">`
  - _Requirements: 1.3, 1.4, 3.1, 3.2, 3.3, 3.4, 6.1, 6.2, 6.3, 6.4, 6.5, 8.1, 8.4, 8.5, 8.6_

- [x] 6. Build `pages/experiments.html`
  - Create `pages/experiments.html`
  - Add the page shell with `<title>Experiments — Unknown Area Navigation</title>`, Google Fonts links, and `<link rel="stylesheet" href="../assets/css/style.css">`
  - Add `<nav class="site-nav">` with five links; mark the Experiments link with `class="nav-active"`
  - Add `<section id="vlm-comparison"><h2>Gemini vs. OpenAI Comparative Experiment</h2>` with:
    - `<h3>Setup</h3>` describing: the two models (Gemini 2.5 Flash Lite and GPT-4.1-nano), the 9-image test set (test1–test10, excluding test4), the exact shared prompt text used for both models, and the timing methodology (wall-clock time per API call)
    - `<h3>Speed Results</h3>` with a `<div class="table-wrapper"><table class="data-table">` with columns Image ID / Gemini (s) / OpenAI (s), 9 data rows (test1–test10 excluding test4) with the exact latency values from the design, plus a summary row showing totals (17.500 s / 17.049 s) and means (1.944 s / 1.894 s)
    - `<h3>Accuracy Results</h3>` with a `<div class="table-wrapper"><table class="data-table">` with columns Image ID / Ground Truth / Gemini Judgment / OpenAI Judgment, 9 data rows with correct/incorrect indicators and `class="result-correct"` or `class="result-incorrect"` on each `<tr>`, plus a summary row showing final accuracy (Gemini: 77.8% (7/9); OpenAI: 88.9% (8/9))
    - `<h3>Error Analysis</h3>` describing: Gemini's over-permissiveness on test1 (morph suit pedestrian), Gemini's over-caution on test7 (graffiti bridge), and OpenAI's over-caution on test9 (roadside hitchhiker)
  - Add `<section id="demo-pipeline"><h2>Demo Pipeline 1</h2>` describing: (a) three-state YOLO confidence classification (below 25% ignored; 25–50% flagged as candidate unknown; ≥ 50% treated as known), (b) Gemini traversability assessment triggered on candidate unknown objects, (c) streak-based confirmation requiring 8 consecutive frames before issuing a verdict, (d) 5-second API cooldown between Gemini calls
  - Add `<footer class="site-footer">`
  - _Requirements: 1.3, 1.4, 3.1, 3.2, 3.3, 3.4, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.4, 8.5, 8.6_

- [x] 7. Checkpoint — verify site structure and navigation
  - Ensure all five HTML files exist at their correct paths (`index.html`, `pages/unknown-object-detection.html`, `pages/multimodal-interpretation.html`, `pages/navigation.html`, `pages/experiments.html`)
  - Ensure `assets/css/style.css` exists
  - Manually open each page in a browser and confirm: the nav bar renders, the active link is visually distinct, all nav links resolve without 404 errors, and no horizontal scrollbar appears at 1024px viewport width
  - Ask the user if any content corrections or design adjustments are needed before writing the tests

- [x] 8. Write property-based tests (`tests/test_site_structure.py`)
  - Create `tests/test_site_structure.py`
  - Add imports: `pytest`, `hypothesis` (`given`, `settings`, `HealthCheck`), `hypothesis.strategies` (`sampled_from`), `bs4.BeautifulSoup`, `pathlib.Path`, `os`
  - Add a module-level `SITE_ROOT` constant pointing to the repository root and a `HTML_FILES` list enumerating all five HTML file paths
  - [x]* 8.1 Write property test for Property 1 — all internal links and asset references use relative paths
    - **Property 1: All internal links and asset references use relative paths**
    - **Validates: Requirements 1.3, 3.4**
    - Strategy: `@given(sampled_from(HTML_FILES))` — for each HTML file, parse with BeautifulSoup, collect all `href` and `src` attribute values, filter out the known external GitHub URL, assert that no remaining value starts with `http://`, `https://`, `/`, or `//`
    - Tag comment: `# Feature: research-website, Property 1: all internal links use relative paths`
    - Use `@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])`
  - [x]* 8.2 Write property test for Property 2 — every page contains required structural elements
    - **Property 2: Every page contains required structural elements**
    - **Validates: Requirements 3.1, 3.2, 3.3, 8.1, 8.6**
    - Strategy: `@given(sampled_from(HTML_FILES))` — for each HTML file, parse with BeautifulSoup and assert: exactly one `<nav class="site-nav">` element; a `<link rel="stylesheet">` whose `href` ends with `assets/css/style.css`; a `<meta name="viewport">` with `content="width=device-width, initial-scale=1"`; nav `<a>` hrefs that resolve to all five destinations (index.html, unknown-object-detection.html, multimodal-interpretation.html, navigation.html, experiments.html); exactly one `<a class="nav-active">` in the nav; the nav-active link's href resolves to the current file's own path
    - Tag comment: `# Feature: research-website, Property 2: every page contains required structural elements`
    - Use `@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])`
  - [x]* 8.3 Write property test for Property 3 — every data table has the `data-table` class
    - **Property 3: Every data table has the data-table class applied**
    - **Validates: Requirement 8.4**
    - Strategy: `@given(sampled_from(HTML_FILES))` — for each HTML file, parse with BeautifulSoup, find all `<table>` elements, assert that every table's class list includes `"data-table"`
    - Tag comment: `# Feature: research-website, Property 3: every data table has data-table class`
    - Use `@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])`

- [x] 9. Final checkpoint — ensure all tests pass
  - Run `pytest tests/test_site_structure.py -v` and confirm all three property tests pass with 100 examples each
  - Ensure all tests pass; ask the user if questions arise

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- All five HTML files share a single stylesheet (`assets/css/style.css`) — task 1 must be completed before any HTML file is opened in a browser for visual verification
- The nav bar HTML is duplicated across all five pages; the active link is set statically per file (no JavaScript required)
- Property tests use `sampled_from(HTML_FILES)` rather than generating random HTML, so Hypothesis exercises each file multiple times across 100 examples — this is the correct strategy for structural invariants over a fixed file set
- The GitHub repository URL used in `index.html` and the footer should be confirmed with the user before publishing; a placeholder is acceptable during development

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2", "3", "4", "5", "6"] },
    { "id": 2, "tasks": ["8.1", "8.2", "8.3"] }
  ]
}
```
