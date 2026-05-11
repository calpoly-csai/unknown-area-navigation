# Design Document — Research Website

## Overview

The research website is a static, multi-page site that presents the Unknown Area Navigation project to an academic and technical audience. It is hosted on GitHub Pages, served directly from the repository root with no build step. The site covers three research subsystems — unknown object detection, multimodal scene interpretation, and AprilTag-based navigation — plus a dedicated experiments page presenting quantitative results.

The stack is plain HTML and CSS with optional vanilla JavaScript. No frameworks, no package manager, no preprocessors. Every page is a self-contained HTML file that links to a single shared stylesheet.

---

## Architecture

The site is a flat collection of HTML files. There is no routing layer, no templating engine, and no JavaScript framework. Navigation between pages is handled by plain `<a href="...">` links with relative paths.

```
unknown-area-navigation/          ← repository root (GitHub Pages root)
├── index.html                    ← Landing page
├── assets/
│   ├── css/
│   │   └── style.css             ← Single shared stylesheet
│   └── images/                   ← Any diagrams or figures (optional)
└── pages/
    ├── unknown-object-detection.html
    ├── multimodal-interpretation.html
    ├── navigation.html
    └── experiments.html
```

GitHub Pages serves the repository root, so `index.html` is served at the root URL (e.g., `https://username.github.io/unknown-area-navigation/`). Sub-pages are served at their literal paths (e.g., `.../pages/unknown-object-detection.html`).

### Relative Path Strategy

All internal links and asset references use relative paths. This is the only approach that works correctly regardless of the GitHub Pages subdomain or repository name.

| From file | To stylesheet | To index | To sub-page |
|---|---|---|---|
| `index.html` | `assets/css/style.css` | `index.html` or `./` | `pages/unknown-object-detection.html` |
| `pages/*.html` | `../assets/css/style.css` | `../index.html` | `../pages/navigation.html` |

No absolute paths (`/assets/...`), no protocol-relative paths, no hardcoded domain names anywhere in the HTML.

---

## Components and Interfaces

### Shared Navigation Bar

Every page includes an identical `<nav>` block. The active-page link is distinguished by adding the CSS class `nav-active` to the `<a>` element that corresponds to the current page.

Because the site has no JavaScript templating, the nav HTML is duplicated across all five pages. The active class is set statically in each file — each page's copy of the nav marks its own link as active.

```html
<nav class="site-nav">
  <a href="../index.html">Home</a>
  <a href="../pages/unknown-object-detection.html">Unknown Object Detection</a>
  <a href="../pages/multimodal-interpretation.html">Multimodal Interpretation</a>
  <a href="../pages/navigation.html">Navigation</a>
  <a href="../pages/experiments.html">Experiments</a>
</nav>
```

On `index.html`, the paths drop the `../` prefix (e.g., `href="index.html"` and `href="pages/unknown-object-detection.html"`).

The active link on each page receives `class="nav-active"` directly in the markup:

```html
<!-- On unknown-object-detection.html -->
<a href="../pages/unknown-object-detection.html" class="nav-active">Unknown Object Detection</a>
```

This approach requires no JavaScript and works with direct URL navigation (no redirect needed).

### Page Shell

Every HTML file follows this shell structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>[Page Title] — Unknown Area Navigation</title>
  <link rel="stylesheet" href="[relative path to assets/css/style.css]">
</head>
<body>
  <nav class="site-nav">
    <!-- nav links, one marked nav-active -->
  </nav>
  <main class="page-content">
    <h1>[Page Heading]</h1>
    <!-- page body -->
  </main>
  <footer class="site-footer">
    <p>Unknown Area Navigation · <a href="https://github.com/[repo]">GitHub</a></p>
  </footer>
</body>
</html>
```

### Tables

All data tables use the class `data-table`. The CSS applies visible borders and alternating row shading via `tbody tr:nth-child(even)`. Tables are wrapped in a `<div class="table-wrapper">` to allow horizontal scrolling on narrow viewports without breaking the page layout.

```html
<div class="table-wrapper">
  <table class="data-table">
    <thead>
      <tr><th>Column A</th><th>Column B</th></tr>
    </thead>
    <tbody>
      <tr><td>...</td><td>...</td></tr>
    </tbody>
  </table>
</div>
```

### Inline Code

All code identifiers, model names, file paths, and parameter names that appear inline in prose are wrapped in `<code>` elements, which the stylesheet renders in a monospace font with a subtle background tint.

---

## Data Models

This is a static site — there is no runtime data model. The "data" is the research content baked into the HTML at authoring time. The relevant structural decisions are:

**Per-page content mapping:**

| File | Requirement | Primary content |
|---|---|---|
| `index.html` | Req 2 | Title, abstract, summary, subsystem overview cards, GitHub link |
| `pages/unknown-object-detection.html` | Req 4 | Problem statement, 4 approaches, parameter tables, limitations |
| `pages/multimodal-interpretation.html` | Req 5 | Task definition, model table, performance observations, utility functions, PBT validation, limitations, experiment results summary |
| `pages/navigation.html` | Req 6 | Task definition, calibration procedure, movement control table, limitations |
| `pages/experiments.html` | Req 7 | Experiment setup, speed results table, accuracy results table, error analysis, Demo Pipeline 1 architecture |

**Table schemas (pre-defined, static):**

*Unknown Object Detection — Key Parameters table:*

| Approach | Parameter Name | Value |
|---|---|---|
| Confidence-Thresholded YOLO-World | Confidence threshold | 50% |
| Dual-Model Consensus | NANO_OWL_THRESHOLD | 0.08 |
| Dual-Model Consensus | YOLO_WORLD_LOW_CONF_THRESHOLD | 0.35 |
| Dual-Model Consensus | IOU_MATCH_THRESHOLD | 0.20 |
| Dual-Model Consensus | INFERENCE_MAX_WIDTH | 640 px |
| Dual-Model Consensus | NANO_OWL_FRAME_INTERVAL | 3 frames |
| Colour Histogram | Histogram bins | 8 per channel |
| Colour Histogram | Change threshold | 0.1 (Bhattacharyya) |
| Colour Histogram | Alert duration | 0.5 s |

*Navigation — Movement Control table:*

| Condition | Evaluated In Order | Command |
|---|---|---|
| `pose_t[0] < −0.075 m` | 1st | Move Left |
| `pose_t[0] > +0.075 m` | 2nd | Move Right |
| `pose_t[2] ≤ 0.1 m` | 3rd | Stop |
| Otherwise | 4th | Move Forward |
| No tag detected | — | No Tag Detected |

*Experiments — Speed Results table:*

| Image | Gemini (s) | OpenAI (s) |
|---|---|---|
| test1 | 1.359 | 1.395 |
| test2 | 2.430 | 1.990 |
| test3 | 2.609 | 2.239 |
| test5 | 3.771 | 3.897 |
| test6 | 1.213 | 1.227 |
| test7 | 1.013 | 0.919 |
| test8 | 1.027 | 0.840 |
| test9 | 2.756 | 1.310 |
| test10 | 1.322 | 3.232 |
| **Total** | **17.500** | **17.049** |
| **Mean** | **1.944** | **1.894** |

*Experiments — Accuracy Results table:*

| Image | Ground Truth | Gemini | OpenAI |
|---|---|---|---|
| test1 | false | ✗ true | ✓ false |
| test2 | false | ✓ false | ✓ false |
| test3 | true | ✓ true | ✓ true |
| test5 | false | ✓ false | ✓ false |
| test6 | false | ✓ false | ✓ false |
| test7 | true | ✗ false | ✓ true |
| test8 | false | ✓ false | ✓ false |
| test9 | true | ✓ true | ✗ false |
| test10 | false | ✓ false | ✓ false |
| **Accuracy** | — | **77.8% (7/9)** | **88.9% (8/9)** |

---

## Visual Design System

### Color Palette

| Role | Value | Usage |
|---|---|---|
| Background | `#0f1117` | Page background |
| Surface | `#1a1d27` | Cards, nav bar, footer |
| Surface raised | `#22263a` | Table header, code blocks |
| Border | `#2e3350` | Table borders, dividers |
| Text primary | `#e8eaf0` | Body text, headings |
| Text secondary | `#8b92b0` | Captions, metadata, footer |
| Accent | `#5b8dee` | Links, active nav, heading accents |
| Accent hover | `#7aa3f5` | Link hover state |
| Success tint | `#1e3a2f` | Correct result rows (experiments table) |
| Error tint | `#3a1e1e` | Incorrect result rows (experiments table) |

The dark background (`#0f1117`) with light text (`#e8eaf0`) achieves a contrast ratio above 12:1, well above the WCAG 2.1 AA minimum of 4.5:1. The accent blue (`#5b8dee` on `#0f1117`) achieves approximately 5.2:1, meeting AA for normal text.

### Typography

```css
font-family: 'Inter', system-ui, -apple-system, sans-serif;
```

Inter is loaded from Google Fonts with a single `<link>` in the `<head>`. The system-ui fallback ensures readable text even if the font fails to load.

| Element | Size | Weight | Line height |
|---|---|---|---|
| `h1` | 2.25rem | 700 | 1.2 |
| `h2` | 1.6rem | 600 | 1.3 |
| `h3` | 1.2rem | 600 | 1.4 |
| Body (`p`, `li`, `td`) | 1rem | 400 | 1.7 |
| `code` | 0.875rem | 400 | — |
| Nav links | 0.9rem | 500 | — |
| Caption / secondary | 0.85rem | 400 | — |

Monospace font for `<code>` elements:

```css
font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
```

Also loaded from Google Fonts with a monospace system fallback.

### Spacing and Layout

The page content is constrained to a maximum width of 860px and centered:

```css
.page-content {
  max-width: 860px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
}
```

This keeps line lengths readable (approximately 70–80 characters at 1rem) without requiring a multi-column layout.

Vertical rhythm uses a base unit of 1.5rem. Section spacing (`h2` top margin) is 3rem. Paragraph spacing is 1rem.

### Navigation Bar

The nav bar is a full-width strip pinned to the top of the viewport:

```css
.site-nav {
  position: sticky;
  top: 0;
  background: #1a1d27;
  border-bottom: 1px solid #2e3350;
  padding: 0 1.5rem;
  display: flex;
  gap: 0;
  z-index: 100;
}

.site-nav a {
  display: inline-block;
  padding: 0.85rem 1.1rem;
  color: #8b92b0;
  text-decoration: none;
  font-size: 0.9rem;
  font-weight: 500;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}

.site-nav a:hover {
  color: #e8eaf0;
}

.site-nav a.nav-active {
  color: #5b8dee;
  border-bottom-color: #5b8dee;
}
```

The active link is distinguished by the accent color and a 2px bottom border. This is a clear visual indicator that does not rely on background color alone (supporting users with color vision deficiencies).

### Tables

```css
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.data-table th {
  background: #22263a;
  color: #e8eaf0;
  font-weight: 600;
  text-align: left;
  padding: 0.65rem 0.9rem;
  border: 1px solid #2e3350;
}

.data-table td {
  padding: 0.6rem 0.9rem;
  border: 1px solid #2e3350;
  color: #c8cce0;
}

.data-table tbody tr:nth-child(even) {
  background: #1a1d27;
}

.data-table tbody tr:hover {
  background: #22263a;
}
```

Result tables on the Experiments page use additional row classes for semantic coloring:

```css
.data-table tr.result-correct { background: #1e3a2f; }
.data-table tr.result-incorrect { background: #3a1e1e; }
```

### Code Blocks

Inline `<code>` elements:

```css
code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.875em;
  background: #22263a;
  color: #a8d8ea;
  padding: 0.15em 0.4em;
  border-radius: 3px;
}
```

### Subsystem Cards (Landing Page)

The three subsystem entries on the landing page are presented as cards in a CSS grid:

```css
.subsystem-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1.25rem;
  margin: 2rem 0;
}

.subsystem-card {
  background: #1a1d27;
  border: 1px solid #2e3350;
  border-radius: 6px;
  padding: 1.25rem 1.5rem;
}

.subsystem-card h3 {
  margin-top: 0;
  color: #5b8dee;
}
```

---

## Per-Page Content Structure

### index.html — Landing Page

```
<nav> (Home active)
<main>
  <h1>Unknown Area Navigation</h1>
  <section id="abstract">
    <h2>Abstract</h2>
    <p>[full abstract text]</p>
  </section>
  <section id="summary">
    <h2>Project Summary</h2>
    <p>[≤150 word summary]</p>
  </section>
  <section id="subsystems">
    <h2>Research Subsystems</h2>
    <div class="subsystem-grid">
      <div class="subsystem-card"> Unknown Object Detection ... </div>
      <div class="subsystem-card"> Multimodal Scene Interpretation ... </div>
      <div class="subsystem-card"> AprilTag-Based Navigation ... </div>
    </div>
  </section>
  <section id="links">
    <a href="https://github.com/[repo]">View on GitHub</a>
  </section>
</main>
<footer>
```

### pages/unknown-object-detection.html

```
<nav> (Unknown Object Detection active)
<main>
  <h1>Unknown Object Detection</h1>
  <section id="problem">
    <h2>Problem Statement</h2>
    <p>[closed-vocabulary limitation, silent mislabelling]</p>
  </section>
  <section id="approaches">
    <h2>Approaches</h2>
    <h3>Approach 1 — Confidence-Thresholded YOLO-World</h3>
    <h3>Approach 2 — Dual-Model Consensus (NanoOWL + YOLO-World)</h3>
    <h3>Approach 3 — Baseline Closed-Vocabulary YOLO</h3>
    <h3>Approach 4 — Frame-to-Frame Colour Histogram Comparison</h3>
  </section>
  <section id="parameters">
    <h2>Key Parameters</h2>
    <div class="table-wrapper">
      <table class="data-table"> [Approach / Parameter / Value] </table>
    </div>
  </section>
  <section id="limitations">
    <h2>Limitations</h2>
    <ul>[qualitative evaluation, CPU NanoOWL, incomplete custom model]</ul>
  </section>
</main>
<footer>
```

### pages/multimodal-interpretation.html

```
<nav> (Multimodal Interpretation active)
<main>
  <h1>Multimodal Scene Interpretation</h1>
  <section id="task">
    <h2>Task Definition</h2>
    <p>[scene description + binary traversability judgment]</p>
  </section>
  <section id="models">
    <h2>Models Evaluated</h2>
    <h3>Local VLMs</h3>
    <table class="data-table"> [Model / Identifier / Parameters] </table>
    <h3>Cloud API VLMs</h3>
    <table class="data-table"> [Model / Provider] </table>
  </section>
  <section id="performance">
    <h2>Performance Observations</h2>
    <p>[10–30+ min local vs 2–5s cloud]</p>
  </section>
  <section id="utilities">
    <h2>Shared Utilities and Correctness Validation</h2>
    <p>[4 helper functions, Hypothesis PBT, 5 properties, 100 examples each]</p>
  </section>
  <section id="experiment-results">
    <h2>Comparative Experiment Results</h2>
    <p>[summary of Gemini vs OpenAI accuracy/latency — links to Experiments page]</p>
  </section>
  <section id="limitations">
    <h2>Limitations</h2>
    <ul>[no ground truth, CPU only, single frames, 2-image test set]</ul>
  </section>
</main>
<footer>
```

### pages/navigation.html

```
<nav> (Navigation active)
<main>
  <h1>AprilTag-Based Navigation</h1>
  <section id="task">
    <h2>Task Definition</h2>
    <p>[detect tag, estimate pose, issue movement command, handle no-tag]</p>
  </section>
  <section id="calibration">
    <h2>Camera Calibration</h2>
    <p>[8×6 checkerboard, 20mm squares, findChessboardCorners, cornerSubPix,
        calibrateCamera, K matrix, distortion coefficients, X/Y/Z convention]</p>
  </section>
  <section id="movement">
    <h2>Movement Control Logic</h2>
    <div class="table-wrapper">
      <table class="data-table"> [Condition / Evaluated In Order / Command] </table>
    </div>
    <p class="table-note">Conditions are evaluated in the order listed...</p>
  </section>
  <section id="limitations">
    <h2>Limitations</h2>
    <ul>[single tag, discrete thresholds, no PID, no temporal smoothing, no integration]</ul>
  </section>
</main>
<footer>
```

### pages/experiments.html

```
<nav> (Experiments active)
<main>
  <h1>Experiments</h1>
  <section id="vlm-comparison">
    <h2>Gemini vs. OpenAI Comparative Experiment</h2>
    <h3>Setup</h3>
    <p>[two models, 9-image test set, shared prompt text, wall-clock timing]</p>
    <h3>Speed Results</h3>
    <div class="table-wrapper">
      <table class="data-table"> [Image / Gemini (s) / OpenAI (s) + summary row] </table>
    </div>
    <h3>Accuracy Results</h3>
    <div class="table-wrapper">
      <table class="data-table"> [Image / Ground Truth / Gemini / OpenAI + accuracy row] </table>
    </div>
    <h3>Error Analysis</h3>
    <p>[Gemini test1, test7 errors; OpenAI test9 error]</p>
  </section>
  <section id="demo-pipeline">
    <h2>Demo Pipeline 1</h2>
    <p>[three-state YOLO classification, streak logic, Gemini assessment, cooldown]</p>
  </section>
</main>
<footer>
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

This site is static HTML/CSS. The correctness properties below are structural invariants that can be verified by parsing the HTML files — they do not require a browser or a running server. They are suitable for property-based testing using a library such as [Hypothesis](https://hypothesis.readthedocs.io/) (Python) combined with an HTML parser such as `BeautifulSoup`, or any equivalent approach that enumerates HTML files and checks structural conditions.

---

### Property 1: All internal links and asset references use relative paths

*For any* HTML file in the site, every `href` and `src` attribute that refers to an internal resource (another page, the stylesheet, or any image) SHALL be a relative path — it must not begin with `http://`, `https://`, `/`, or `//`.

**Validates: Requirements 1.3, 3.4**

---

### Property 2: Every page contains the required structural elements

*For any* HTML file in the site, the file SHALL contain:
- exactly one `<nav class="site-nav">` element,
- a `<link rel="stylesheet">` element whose `href` resolves to `assets/css/style.css` via the relative path from that file's location,
- a `<meta name="viewport">` tag with `content="width=device-width, initial-scale=1"`,
- links in the nav to all five destinations: `index.html`, `unknown-object-detection.html`, `multimodal-interpretation.html`, `navigation.html`, and `experiments.html`,
- exactly one nav link with `class="nav-active"`, and that link SHALL correspond to the current page.

**Validates: Requirements 3.1, 3.2, 3.3, 8.1, 8.6**

---

### Property 3: Every data table has the styled-table class applied

*For any* `<table>` element in any HTML file in the site, the element SHALL have `class="data-table"` (or a class list that includes `data-table`), ensuring that the shared stylesheet's border and alternating-row rules are applied to all tabular data.

**Validates: Requirement 8.4**

---

## Error Handling

Static sites have a narrow error surface. The relevant failure modes and mitigations are:

| Failure | Cause | Mitigation |
|---|---|---|
| 404 on internal link | Absolute path used instead of relative | All paths are relative; verified by Property 1 |
| 404 on stylesheet | Wrong relative depth from `pages/` | `../assets/css/style.css` from all sub-pages; verified by Property 2 |
| Broken active-nav state | Wrong page marked active | Each page's nav is authored independently with the correct link marked |
| Table unstyled | Missing `data-table` class | Verified by Property 3 |
| Layout breaks on mobile | Missing viewport meta | Verified by Property 2 |
| Font fails to load | Google Fonts unavailable | System-ui and monospace fallbacks defined in CSS |
| GitHub Pages 404 on repo rename | Hardcoded repo name in paths | No hardcoded repo names; all paths are relative |

---

## Testing Strategy

This feature is a static HTML/CSS site. Property-based testing (PBT) is applicable to the structural invariants of the HTML files — these are pure functions of the file content and can be tested with 100+ iterations across all pages and elements.

### Property-Based Tests

Use **Hypothesis** (Python) with **BeautifulSoup4** for HTML parsing.

Each property test enumerates all HTML files in the site (or generates representative file content) and asserts the structural invariant.

**Test configuration:**
- Minimum 100 iterations per property test (Hypothesis default)
- Each test references its design property via a comment tag: `# Feature: research-website, Property N: <property text>`

**Property 1 test — Relative paths:**
- Strategy: enumerate all `href` and `src` attributes across all HTML files
- Assert: no value starts with `http://`, `https://`, `/`, or `//` (except the GitHub external link, which is explicitly excluded)
- Tag: `# Feature: research-website, Property 1: all internal links use relative paths`

**Property 2 test — Per-page structural invariants:**
- Strategy: for each HTML file, parse with BeautifulSoup and check all required elements
- Assert: nav present, stylesheet link present and correct, viewport meta present, all five nav links present, exactly one nav-active link, nav-active link matches current page
- Tag: `# Feature: research-website, Property 2: every page contains required structural elements`

**Property 3 test — Table styling:**
- Strategy: for each HTML file, find all `<table>` elements
- Assert: every table has `data-table` in its class list
- Tag: `# Feature: research-website, Property 3: every data table has data-table class`

### Unit / Example Tests

- Verify `index.html` contains an `<h1>` with text "Unknown Area Navigation"
- Verify the abstract section contains the expected opening sentence
- Verify the summary section word count is ≤ 150
- Verify the GitHub repository link is present on the landing page
- Verify the speed results table on `experiments.html` has 11 rows (9 images + header + summary)
- Verify the accuracy results table has the correct final accuracy values (77.8%, 88.9%)
- Verify the movement control table on `navigation.html` has 5 data rows

### Manual / Visual Tests

- Render each page in a browser at 1024px viewport width and confirm no horizontal scrollbar
- Verify WCAG 2.1 AA contrast ratios using a browser extension (e.g., axe DevTools)
- Confirm sticky nav bar remains visible when scrolling long pages
- Confirm Google Fonts load correctly; confirm fallback fonts are readable if fonts are blocked
- Confirm the active nav link is visually distinct on each page
