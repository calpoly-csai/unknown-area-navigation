# Multimodal Interpretation Experiment: Gemini vs. OpenAI

## Overview

This experiment evaluates two cloud-based vision-language models (VLMs) on their ability to assess road traversability for an autonomous navigation system. Both models receive the same prompt and the same set of test images, and are asked to return a structured JSON response containing a scene description, a binary traversability judgment, and a one-sentence justification.

**Models evaluated:**
- `gemini-2.5-flash-lite` (Google Gemini API)
- `gpt-4.1-nano` (OpenAI API)

**Test set:** 9 images (test1–test10, excluding test4 which was unavailable)

---

## Methodology

Each model was called sequentially via its respective API using the same prompt:

> *"Analyse this image for an autonomous vehicle. Reply with a JSON object with exactly these fields: `description`, `traversable` (true/false), `justification`. Return only the raw JSON, no markdown fences."*

Wall-clock time was measured per API call using `time.perf_counter()`. Accuracy was assessed manually by the researcher based on whether the traversability judgment (`true`/`false`) was correct for each image.

---

## Speed Results

| Image    | Gemini (s) | OpenAI (s) |
|----------|-----------|-----------|
| test1    | 1.359     | 1.395     |
| test2    | 2.430     | 1.990     |
| test3    | 2.609     | 2.239     |
| test5    | 3.771     | 3.897     |
| test6    | 1.213     | 1.227     |
| test7    | 1.013     | 0.919     |
| test8    | 1.027     | 0.840     |
| test9    | 2.756     | 1.310     |
| test10   | 1.322     | 3.232     |
| **Total**    | **17.500** | **17.049** |
| **Mean** | **1.944** | **1.894** |

Both models were comparable in speed, with OpenAI averaging **1.894s** per image versus Gemini's **1.944s** — a difference of roughly 50ms. Neither model showed a consistent latency advantage across all images; Gemini was faster on test10 (1.322s vs. 3.232s) while OpenAI was faster on test9 (1.310s vs. 2.756s).

---

## Accuracy Results

Accuracy is defined as whether the model's `traversable` judgment matched the ground truth for each image.

| Image  | Ground Truth | Gemini        | OpenAI        |
|--------|-------------|---------------|---------------|
| test1  | false       | ✗ true        | ✓ false       |
| test2  | false       | ✓ false       | ✓ false       |
| test3  | true        | ✓ true        | ✓ true        |
| test5  | false       | ✓ false       | ✓ false       |
| test6  | false       | ✓ false       | ✓ false       |
| test7  | true        | ✗ false       | ✓ true        |
| test8  | false       | ✓ false       | ✓ false       |
| test9  | true        | ✓ true        | ✗ false       |
| test10 | false       | ✓ false       | ✓ false       |
| **Correct** | —      | **7 / 9**     | **8 / 9**     |
| **Accuracy** | —     | **77.8%**     | **88.9%**     |

### Error Analysis

**Gemini errors:**

- **test1** — A person in a morph suit standing next to a novelty pedestrian sign. Gemini judged the scene traversable, reasoning that the signage and person did not obstruct the road. The model failed to recognise the pedestrian-adjacent hazard as a reason to stop.
- **test7** — A graffiti-covered bridge. Gemini judged it non-traversable, citing structural concerns based on the graffiti. The bridge was in fact passable; the model over-indexed on visual degradation as a proxy for structural unsafety.

**OpenAI errors:**

- **test9** — A woman hitchhiking on the roadside with a clear road ahead. OpenAI judged the scene non-traversable, citing the woman's proximity to the road as a hazard. The road itself was clear and the correct judgment was traversable.

---

## Summary

| Metric           | Gemini 2.5 Flash Lite | GPT-4.1-nano |
|------------------|-----------------------|--------------|
| Accuracy         | 77.8% (7/9)           | 88.9% (8/9)  |
| Mean latency     | 1.944s                | 1.894s       |
| Total time (9 images) | 17.500s          | 17.049s      |
| Errors           | test1, test7          | test9        |

GPT-4.1-nano outperformed Gemini 2.5 Flash Lite on this task, achieving higher traversability accuracy (88.9% vs. 77.8%) at comparable speed. Gemini's errors suggest a tendency toward over-permissiveness in ambiguous pedestrian scenes (test1) and over-caution based on visual surface features (test7). OpenAI's single error reflects over-caution in a scene with a roadside pedestrian. Both models performed well on clear-cut cases (fallen tree, flooded road, livestock, aircraft on road) and struggled only on edge cases involving human presence or ambiguous infrastructure.

Given the near-identical latency profiles, accuracy is the primary differentiator for this use case, favouring GPT-4.1-nano for deployment in traversability assessment pipelines.
