# Requirements Document

## Introduction

This feature covers the design and implementation of a static research website for the **Unknown Area Navigation** project, hosted on GitHub Pages. The website presents the project's research contributions — unknown object detection, multimodal scene interpretation, and AprilTag-based navigation — to an academic and technical audience. It serves as the public-facing summary of the research: its motivation, methods, experiments, results, and findings.

The site consists of a root `index.html` landing page and a set of sub-pages under a `pages/` folder, all written in plain HTML/CSS with no build toolchain required, so the site deploys directly from the repository root on GitHub Pages.

---

## Glossary

- **Website**: The complete static GitHub Pages site for the Unknown Area Navigation research project.
- **Landing_Page**: The root `index.html` file served at the repository's GitHub Pages URL.
- **Sub_Page**: Any HTML file located under the `pages/` folder.
- **Navigation_Bar**: The persistent header element present on every page containing links to all major sections.
- **Abstract_Section**: The section of the Landing_Page that presents the project abstract.
- **Methods_Section**: A Sub_Page or section summarising the three research subsystems.
- **Unknown_Object_Detection_Page**: The Sub_Page dedicated to the unknown object detection subsystem.
- **Multimodal_Interpretation_Page**: The Sub_Page dedicated to the multimodal scene interpretation subsystem.
- **Navigation_Page**: The Sub_Page dedicated to the AprilTag-based navigation subsystem.
- **Experiments_Page**: The Sub_Page presenting the comparative VLM experiment and the end-to-end demo pipeline.
- **Results_Section**: The portion of the Experiments_Page that presents quantitative accuracy and latency results.
- **Visitor**: Any person who loads the Website in a web browser.

---

## Requirements

### Requirement 1: Site Structure and GitHub Pages Compatibility

**User Story:** As a researcher, I want the website to be deployable directly from the repository root on GitHub Pages, so that no build step or external toolchain is needed.

#### Acceptance Criteria

1. THE Website SHALL consist of a root `index.html` file and all Sub_Pages located under a `pages/` folder.
2. THE Website SHALL use only static HTML, CSS, and optionally vanilla JavaScript — no server-side code, no build tools, and no external frameworks that require a package manager.
3. THE Website SHALL use relative paths for all internal links and asset references so that, when served from any GitHub Pages subdomain, the browser returns no HTTP 4xx errors for any internal link or asset request.
4. IF a Visitor navigates to a Sub_Page URL directly, THEN THE Website SHALL render that Sub_Page with all assets (CSS, images, scripts) loaded by the browser without requiring a redirect through the Landing_Page.

---

### Requirement 2: Landing Page

**User Story:** As a Visitor, I want a clear and informative landing page, so that I can immediately understand what the research is about and navigate to any section.

#### Acceptance Criteria

1. THE Landing_Page SHALL display the project title "Unknown Area Navigation" as the primary heading.
2. THE Landing_Page SHALL include an Abstract_Section that presents the full project abstract sourced from the project documentation.
3. THE Landing_Page SHALL include a project summary section of no more than 150 words that states: (a) that deep learning models trained on fixed datasets fail to generalise to out-of-distribution environments, and (b) that this project develops a supplementary pipeline to address that limitation.
4. THE Landing_Page SHALL include a Navigation_Bar with links to the Unknown_Object_Detection_Page, the Multimodal_Interpretation_Page, the Navigation_Page, and the Experiments_Page.
5. THE Landing_Page SHALL include a section listing the three research subsystems (Unknown Object Detection, Multimodal Scene Interpretation, AprilTag-Based Navigation) with a one-sentence description of each and a link to the corresponding Sub_Page.
6. THE Landing_Page SHALL include a link to the project's GitHub repository.

---

### Requirement 3: Navigation Bar

**User Story:** As a Visitor, I want a consistent navigation bar on every page, so that I can move between sections without using the browser back button.

#### Acceptance Criteria

1. THE Navigation_Bar SHALL appear on the Landing_Page and on every Sub_Page.
2. THE Navigation_Bar SHALL contain links to: the Landing_Page (Home), the Unknown_Object_Detection_Page, the Multimodal_Interpretation_Page, the Navigation_Page, and the Experiments_Page.
3. WHILE a Visitor is viewing a given page, THE Navigation_Bar SHALL render the link corresponding to that page in a distinct visual style (e.g., different background colour, font weight, or underline) that differs from the style of inactive links.
4. THE Navigation_Bar SHALL use relative paths so that clicking any Navigation_Bar link navigates the browser to the target page without producing a 404 error, regardless of which GitHub Pages subdomain serves the site.

---

### Requirement 4: Unknown Object Detection Sub-Page

**User Story:** As a Visitor, I want a dedicated page for the unknown object detection subsystem, so that I can understand the problem, the approaches taken, and their trade-offs.

#### Acceptance Criteria

1. THE Unknown_Object_Detection_Page SHALL include a problem statement section that explicitly states: (a) that standard closed-vocabulary detectors assign the nearest known class label to out-of-distribution objects, and (b) that this silent mislabelling is the core problem the subsystem addresses.
2. THE Unknown_Object_Detection_Page SHALL describe all four implemented approaches — Confidence-Thresholded YOLO-World, Dual-Model Consensus (NanoOWL + YOLO-World), Baseline Closed-Vocabulary YOLO, and Frame-to-Frame Colour Histogram Comparison — each with a minimum of: (a) a one-sentence description of the approach, (b) the detection or flagging mechanism used, and (c) the intended use case or trade-off.
3. THE Unknown_Object_Detection_Page SHALL present the key parameters for each approach in an HTML `<table>` element with at minimum the columns: Approach, Parameter Name, and Value — populated with the confidence thresholds, IoU thresholds, and histogram distance thresholds used in the implementation.
4. THE Unknown_Object_Detection_Page SHALL include a limitations section that explicitly states: (a) that evaluation was qualitative with no quantitative benchmark, (b) that NanoOWL was run on CPU rather than its intended NVIDIA Jetson edge hardware, and (c) that a planned custom class-agnostic detection model was not completed.
5. THE Unknown_Object_Detection_Page SHALL include the Navigation_Bar.

---

### Requirement 5: Multimodal Scene Interpretation Sub-Page

**User Story:** As a Visitor, I want a dedicated page for the multimodal interpretation subsystem, so that I can understand how VLMs were used to assess scene traversability.

#### Acceptance Criteria

1. THE Multimodal_Interpretation_Page SHALL describe the task: given a single image, produce a natural-language scene description and a binary traversability judgment (safe to pass / not safe to pass).
2. THE Multimodal_Interpretation_Page SHALL list all six models evaluated in a structured format, distinguishing between local VLMs (Qwen2-VL, LLaVA-1.5, Moondream2, InternVL2-2B) and cloud API VLMs (GPT-4o-mini, Gemini 2.0 Flash Lite); for each local VLM the listing SHALL include its parameter count, and for each cloud API VLM the listing SHALL include its provider name.
3. THE Multimodal_Interpretation_Page SHALL present the performance observation that local VLMs required 10–30+ minutes per image on CPU, while cloud API models returned results in 2–5 seconds.
4. THE Multimodal_Interpretation_Page SHALL describe the four shared utility functions (`parse_traversability`, `build_output`, `get_output_path`, `is_jpg`) and state that each was validated using property-based testing with the Hypothesis library against a minimum of 5 correctness properties, each tested against at least 100 randomly generated examples.
5. THE Multimodal_Interpretation_Page SHALL include a limitations section covering: no ground-truth labels, CPU-only inference, single static frames with no temporal context, and a non-representative two-image test set.
6. THE Multimodal_Interpretation_Page SHALL include a section presenting the results of the comparative VLM experiment (accuracy and latency), consistent with the data shown on the Experiments_Page.
7. THE Multimodal_Interpretation_Page SHALL include the Navigation_Bar.

---

### Requirement 6: AprilTag Navigation Sub-Page

**User Story:** As a Visitor, I want a dedicated page for the navigation subsystem, so that I can understand how AprilTag-based pose estimation was used for directional control.

#### Acceptance Criteria

1. THE Navigation_Page SHALL describe the task as: detect an AprilTag in a live camera feed, estimate its 3D pose using `cv2.solvePnP`, and issue one of five movement states — Move Left, Move Right, Move Forward, Stop, and No Tag Detected — where "No Tag Detected" is the state emitted when no AprilTag is visible in the current frame.
2. THE Navigation_Page SHALL describe the camera calibration procedure, including: (a) the checkerboard target dimensions (number of inner corners and square size in mm), (b) the use of `cv2.findChessboardCorners` and `cv2.cornerSubPix` for corner detection, (c) the use of `cv2.calibrateCamera` to compute the camera matrix K and distortion coefficients, and (d) the coordinate convention used (X right, Y down, Z into the scene).
3. THE Navigation_Page SHALL present the movement control logic in an HTML `<table>` element with the columns: Condition, Evaluated In Order, and Command — populated with: (a) lateral offset > +7.5 cm → Move Right, (b) lateral offset < −7.5 cm → Move Left, (c) distance ≤ 10 cm → Stop, (d) otherwise → Move Forward; and the table SHALL include a note that conditions are evaluated in the order listed.
4. THE Navigation_Page SHALL include a limitations section noting: single-tag assumption, discrete threshold-based commands with no PID controller, no temporal smoothing, and no integration with the other two subsystems.
5. THE Navigation_Page SHALL include the Navigation_Bar.

---

### Requirement 7: Experiments Sub-Page

**User Story:** As a Visitor, I want a dedicated experiments page, so that I can review the quantitative results of the VLM comparison and the end-to-end demo pipeline.

#### Acceptance Criteria

1. THE Experiments_Page SHALL describe the Gemini vs. OpenAI comparative experiment, including: (a) the two models evaluated (Gemini 2.5 Flash Lite and GPT-4.1-nano), (b) the 9-image test set (test1–test10, excluding test4), (c) the exact shared prompt text used for both models, and (d) the timing methodology (wall-clock time per API call).
2. THE Experiments_Page SHALL present the speed results in an HTML `<table>` with rows for each test image (test1–test10, excluding test4) and columns for: Image ID, Gemini latency (s), OpenAI latency (s); plus a summary row showing total time and mean latency (Gemini: 1.944 s mean; OpenAI: 1.894 s mean).
3. THE Experiments_Page SHALL present the accuracy results in an HTML `<table>` with rows for each test image (test1–test10, excluding test4) and columns for: Image ID, Ground Truth, Gemini Judgment, OpenAI Judgment; plus a summary row showing final accuracy scores (Gemini: 77.8%; OpenAI: 88.9%).
4. THE Experiments_Page SHALL include an error analysis section describing the specific failure cases: Gemini's over-permissiveness on test1 (morph suit pedestrian) and over-caution on test7 (graffiti bridge), and OpenAI's over-caution on test9 (roadside hitchhiker).
5. THE Experiments_Page SHALL describe the Demo Pipeline 1 architecture, including: (a) the three-state YOLO confidence classification (detections below 25% confidence are ignored; detections between 25–50% are flagged as candidate unknown objects; detections at or above 50% are treated as known objects), (b) the Gemini traversability assessment triggered on candidate unknown objects, (c) the streak-based confirmation logic requiring 8 consecutive frames before issuing a traversability verdict, and (d) the 5-second API cooldown between Gemini calls.
6. THE Experiments_Page SHALL include the Navigation_Bar.

---

### Requirement 8: Visual Design and Readability

**User Story:** As a Visitor, I want the website to be visually clean and easy to read, so that I can focus on the research content without distraction.

#### Acceptance Criteria

1. THE Website SHALL use the same font families and heading hierarchy (h1, h2, h3 sizes and weights) across all pages, defined in a single shared CSS file or `<style>` block included on every page.
2. THE Website SHALL render without horizontal scrollbars and with all content visible and readable at desktop viewport widths of 1024px and above.
3. THE Website SHALL use sufficient color contrast between text and background to meet WCAG 2.1 AA contrast ratio requirements (minimum 4.5:1 for normal text).
4. THE Website SHALL present tabular data (parameter tables, results tables) using HTML `<table>` elements with visible borders or alternating row shading.
5. WHERE code identifiers, model names, or file paths appear inline in prose, THE Website SHALL render them in a monospace font using `<code>` elements.
6. THE Website SHALL include a `<meta name="viewport" content="width=device-width, initial-scale=1">` tag on every page so that the layout does not break on mobile viewports narrower than 1024px.
