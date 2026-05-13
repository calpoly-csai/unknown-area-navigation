import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import sampled_from
from bs4 import BeautifulSoup
from pathlib import Path
import os

SITE_ROOT = Path("/Users/ivantorriani/Documents/Chapter-III-2026/projects/CSAI Research/unknown-area-navigation")

HTML_FILES = [
    SITE_ROOT / "index.html",
    SITE_ROOT / "pages" / "unknown-object-detection.html",
    SITE_ROOT / "pages" / "multimodal-interpretation.html",
    SITE_ROOT / "pages" / "navigation.html",
    SITE_ROOT / "pages" / "experiments.html",
]

# The five destination basenames every nav must link to
NAV_DESTINATIONS = {
    "index.html",
    "unknown-object-detection.html",
    "multimodal-interpretation.html",
    "navigation.html",
    "experiments.html",
}

# Known intentional external resources that are allowed
ALLOWED_EXTERNAL = {
    "https://github.com/ivantorriani/unknown-area-navigation",
    "https://fonts.googleapis.com",
    "https://fonts.gstatic.com",
}


def is_allowed_external(value: str) -> bool:
    """Return True if the URL is a known, intentional external resource."""
    for allowed in ALLOWED_EXTERNAL:
        if value.startswith(allowed):
            return True
    return False


# Feature: research-website, Property 1: all internal links use relative paths
@given(sampled_from(HTML_FILES))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_all_internal_links_are_relative(html_file: Path):
    """
    **Validates: Requirements 1.3, 3.4**

    Property 1: All internal links and asset references use relative paths.

    For any HTML file in the site, every href and src attribute that refers to
    an internal resource SHALL be a relative path — it must not begin with
    http://, https://, /, or //.

    Known intentional external resources (GitHub repo link, Google Fonts) are
    explicitly excluded from this check.
    """
    assert html_file.exists(), f"HTML file not found: {html_file}"

    content = html_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "html.parser")

    absolute_path_prefixes = ("http://", "https://", "//", "/")

    violations = []

    for tag in soup.find_all(True):
        for attr in ("href", "src"):
            value = tag.get(attr)
            if value is None:
                continue
            value = value.strip()
            if not value:
                continue
            # Skip known intentional external resources
            if is_allowed_external(value):
                continue
            # Check for absolute path patterns
            for prefix in absolute_path_prefixes:
                if value.startswith(prefix):
                    violations.append(
                        f"[{html_file.name}] <{tag.name} {attr}=\"{value}\"> uses an absolute path"
                    )
                    break

    assert violations == [], (
        f"Found {len(violations)} absolute path(s) in {html_file.name}:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# Feature: research-website, Property 2: every page contains required structural elements
@given(sampled_from(HTML_FILES))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_every_page_has_required_structural_elements(html_file):
    """
    **Validates: Requirements 3.1, 3.2, 3.3, 8.1, 8.6**

    For every HTML file in the site:
    1. Exactly one <nav class="site-nav"> element.
    2. A <link rel="stylesheet"> whose href ends with assets/css/style.css.
    3. A <meta name="viewport"> with content="width=device-width, initial-scale=1".
    4. Nav <a> hrefs that resolve to all five destinations (by basename).
    5. Exactly one <a class="nav-active"> in the nav.
    6. The nav-active link's href resolves to the current file's own path (by basename).
    """
    content = html_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "html.parser")

    # 1. Exactly one <nav class="site-nav">
    navs = soup.find_all("nav", class_="site-nav")
    assert len(navs) == 1, (
        f"{html_file.name}: expected exactly 1 <nav class='site-nav'>, found {len(navs)}"
    )
    nav = navs[0]

    # 2. A <link rel="stylesheet"> whose href ends with assets/css/style.css
    stylesheet_links = soup.find_all("link", rel="stylesheet")
    matching_stylesheets = [
        lnk for lnk in stylesheet_links
        if lnk.get("href", "").endswith("assets/css/style.css")
    ]
    assert len(matching_stylesheets) >= 1, (
        f"{html_file.name}: no <link rel='stylesheet'> with href ending in 'assets/css/style.css'"
    )

    # 3. A <meta name="viewport"> with content="width=device-width, initial-scale=1"
    viewport_metas = soup.find_all("meta", attrs={"name": "viewport"})
    assert len(viewport_metas) >= 1, (
        f"{html_file.name}: no <meta name='viewport'> found"
    )
    viewport_contents = [m.get("content", "") for m in viewport_metas]
    assert "width=device-width, initial-scale=1" in viewport_contents, (
        f"{html_file.name}: <meta name='viewport'> does not have "
        f"content='width=device-width, initial-scale=1'. Found: {viewport_contents}"
    )

    # 4. Nav <a> hrefs resolve to all five destinations (check by basename)
    nav_links = nav.find_all("a")
    nav_basenames = {Path(a.get("href", "")).name for a in nav_links}
    missing = NAV_DESTINATIONS - nav_basenames
    assert not missing, (
        f"{html_file.name}: nav is missing links to: {missing}. "
        f"Found basenames: {nav_basenames}"
    )

    # 5. Exactly one <a class="nav-active"> in the nav
    active_links = nav.find_all("a", class_="nav-active")
    assert len(active_links) == 1, (
        f"{html_file.name}: expected exactly 1 <a class='nav-active'> in nav, "
        f"found {len(active_links)}"
    )

    # 6. The nav-active link's href resolves to the current file's own basename
    active_href = active_links[0].get("href", "")
    active_basename = Path(active_href).name
    current_basename = html_file.name
    assert active_basename == current_basename, (
        f"{html_file.name}: nav-active href basename is '{active_basename}', "
        f"expected '{current_basename}'"
    )


# Feature: research-website, Property 3: every data table has data-table class
@given(sampled_from(HTML_FILES))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_every_data_table_has_data_table_class(html_file):
    """
    **Validates: Requirement 8.4**

    For every HTML file in the site, every <table> element SHALL have
    "data-table" in its class list, ensuring the shared stylesheet's border
    and alternating-row rules are applied to all tabular data.
    """
    content = html_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "html.parser")

    tables = soup.find_all("table")
    for table in tables:
        classes = table.get("class") or []
        assert "data-table" in classes, (
            f"{html_file.name}: found a <table> without the 'data-table' class. "
            f"Classes found: {classes}"
        )
