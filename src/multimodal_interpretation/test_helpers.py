"""
Property-based tests for src/multimodal_interpretation/helpers.py.

Uses hypothesis to verify the five correctness properties defined in the design doc.

**Validates: Requirements 1.3, 3.4, 4.2, 4.3, 5.1, 5.2**
"""

import os
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from helpers import build_output, get_output_path, is_jpg, parse_traversability

# ---------------------------------------------------------------------------
# Helpers / generators
# ---------------------------------------------------------------------------

# Any printable text that does NOT start with "yes" (case-insensitive)
non_yes_text = st.text(min_size=0).filter(
    lambda s: not s.strip().lower().startswith("yes")
)

# Strings that start with some case variant of "yes"
yes_variants = st.sampled_from(
    ["yes", "Yes", "YES", "yEs", "yES", "YeS", "YEs", "YESsir", "yes please", "Yes!"]
)

# Arbitrary text with optional leading/trailing whitespace
padded_text = st.builds(
    lambda s, pre, post: pre + s + post,
    s=st.text(),
    pre=st.text(alphabet=string.whitespace, min_size=0, max_size=5),
    post=st.text(alphabet=string.whitespace, min_size=0, max_size=5),
)

# Simple file-path-like strings: directory + filename + extension
_path_component = st.text(
    alphabet=st.characters(blacklist_characters="/\\\x00"), min_size=1, max_size=20
)
_extension = st.text(
    alphabet=st.characters(blacklist_characters="/\\\x00."), min_size=1, max_size=5
)
arbitrary_image_path = st.builds(
    lambda d, f, e: os.path.join(d, f"{f}.{e}"),
    d=_path_component,
    f=_path_component,
    e=_extension,
)

# Extensions that are NOT ".jpg" (case-insensitive)
non_jpg_extension = st.text(
    alphabet=st.characters(blacklist_characters="/\\\x00."), min_size=0, max_size=10
).filter(lambda e: e.lower() != "jpg")


# ---------------------------------------------------------------------------
# Property 1 — parse_traversability is True iff response starts with "yes"
# **Validates: Requirements 4.2, 4.3**
# ---------------------------------------------------------------------------


@given(response=yes_variants)
@settings(max_examples=100)
def test_property1_yes_prefix_returns_true(response):
    """Property 1 (yes branch): parse_traversability returns True for yes-prefixed strings."""
    assert parse_traversability(response) is True


@given(response=non_yes_text)
@settings(max_examples=100)
def test_property1_non_yes_returns_false(response):
    """Property 1 (non-yes branch): parse_traversability returns False for all other strings."""
    assert parse_traversability(response) is False


# ---------------------------------------------------------------------------
# Property 2 — description whitespace is stripped
# **Validates: Requirements 3.4**
# ---------------------------------------------------------------------------


@given(description=padded_text, image_path=arbitrary_image_path, traversability=st.booleans())
@settings(max_examples=100)
def test_property2_description_is_stripped(description, image_path, traversability):
    """Property 2: imageDescription in build_output equals description.strip()."""
    result = build_output(image_path, description, traversability)
    assert result["imageDescription"] == description.strip()


# ---------------------------------------------------------------------------
# Property 3 — build_output always produces exactly three fields
# **Validates: Requirements 5.2**
# ---------------------------------------------------------------------------


@given(
    image_path=arbitrary_image_path,
    description=st.text(),
    traversability=st.booleans(),
)
@settings(max_examples=100)
def test_property3_build_output_exact_keys(image_path, description, traversability):
    """Property 3: build_output key set is exactly {imageName, imageDescription, traversability}."""
    result = build_output(image_path, description, traversability)
    assert set(result.keys()) == {"imageName", "imageDescription", "traversability"}


# ---------------------------------------------------------------------------
# Property 4 — output path is co-located with input image
# **Validates: Requirements 5.1**
# ---------------------------------------------------------------------------


@given(image_path=arbitrary_image_path)
@settings(max_examples=100)
def test_property4_output_path_colocation(image_path):
    """Property 4: get_output_path returns a path in the same directory as image_path."""
    output_path = get_output_path(image_path)
    assert os.path.dirname(output_path) == os.path.dirname(image_path)


@given(image_path=arbitrary_image_path)
@settings(max_examples=100)
def test_property4_output_path_naming(image_path):
    """Property 4: get_output_path filename is {base}_output.json."""
    output_path = get_output_path(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]
    assert os.path.basename(output_path) == f"{base}_output.json"


# ---------------------------------------------------------------------------
# Property 5 — non-.jpg extensions are always rejected by is_jpg
# **Validates: Requirements 1.3**
# ---------------------------------------------------------------------------


@given(
    directory=_path_component,
    filename=_path_component,
    ext=non_jpg_extension,
)
@settings(max_examples=100)
def test_property5_non_jpg_rejected(directory, filename, ext):
    """Property 5: is_jpg returns False for any extension that is not .jpg (case-insensitive)."""
    path = os.path.join(directory, f"{filename}.{ext}")
    assert is_jpg(path) is False


@pytest.mark.parametrize("path", ["image.jpg", "image.JPG", "image.Jpg", "/some/dir/photo.JPG"])
def test_property5_jpg_accepted(path):
    """Property 5 (positive): is_jpg returns True for .jpg extensions (any case)."""
    assert is_jpg(path) is True
