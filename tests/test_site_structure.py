from bs4 import BeautifulSoup
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parent.parent

INDEX_HTML = SITE_ROOT / "index.html"

REQUIRED_STYLESHEETS = {
    "assets/css/bulma.min.css",
    "assets/css/nerfies.css",
    "assets/css/portfolio.css",
}

# Known intentional external resources that are allowed
ALLOWED_EXTERNAL = {
    "https://cdnjs.cloudflare.com",
    "https://fonts.googleapis.com",
    "https://fonts.gstatic.com",
    "https://github.com/calpoly-csai",
    "https://www.linkedin.com/in/",
}


def is_allowed_external(value: str) -> bool:
    """Return True if the URL is a known, intentional external resource."""
    return any(value.startswith(allowed) for allowed in ALLOWED_EXTERNAL)


# Feature: research-portfolio, Property 1: all internal links use relative paths
def test_all_internal_links_are_relative():
    """
    For the single-page site, every href and src attribute that refers to an
    internal resource SHALL be a relative path — it must not begin with
    http://, https://, /, or //.

    Known intentional external resources (GitHub org/repo, LinkedIn profiles,
    Google Fonts, cdnjs Font Awesome) are explicitly excluded from this check.
    """
    assert INDEX_HTML.exists(), f"HTML file not found: {INDEX_HTML}"

    content = INDEX_HTML.read_text(encoding="utf-8")
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
            if is_allowed_external(value):
                continue
            for prefix in absolute_path_prefixes:
                if value.startswith(prefix):
                    violations.append(
                        f"<{tag.name} {attr}=\"{value}\"> uses an absolute path"
                    )
                    break

    assert violations == [], (
        f"Found {len(violations)} absolute path(s) in index.html:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_required_structural_elements():
    """
    index.html SHALL contain:
    1. A <title>.
    2. A <meta name="viewport"> with content="width=device-width, initial-scale=1".
    3. <link rel="stylesheet"> entries whose hrefs cover bulma.min.css, nerfies.css,
       and portfolio.css.
    """
    content = INDEX_HTML.read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "html.parser")

    title = soup.find("title")
    assert title is not None and title.text.strip(), "index.html is missing a <title>"

    viewport_metas = soup.find_all("meta", attrs={"name": "viewport"})
    viewport_contents = [m.get("content", "") for m in viewport_metas]
    assert "width=device-width, initial-scale=1" in viewport_contents, (
        f"<meta name='viewport'> missing or incorrect. Found: {viewport_contents}"
    )

    stylesheet_hrefs = {
        lnk.get("href", "") for lnk in soup.find_all("link", rel="stylesheet")
    }
    missing_stylesheets = {
        required
        for required in REQUIRED_STYLESHEETS
        if not any(href.endswith(required) for href in stylesheet_hrefs)
    }
    assert not missing_stylesheets, (
        f"index.html is missing stylesheet link(s): {missing_stylesheets}"
    )


# Feature: research-portfolio, Property 2: every table uses Bulma's table class
def test_every_table_has_bulma_table_class():
    """
    Every <table> element in index.html SHALL have "table" in its class list,
    ensuring Bulma's table styling is applied to all tabular data.
    """
    content = INDEX_HTML.read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "html.parser")

    tables = soup.find_all("table")
    assert tables, "expected at least one <table> in index.html"

    for table in tables:
        classes = table.get("class") or []
        assert "table" in classes, (
            f"found a <table> without Bulma's 'table' class. Classes found: {classes}"
        )


def test_all_referenced_local_assets_exist():
    """
    Every relative href/src in index.html SHALL resolve to a file that
    actually exists on disk, so the published page never links to a
    missing asset.
    """
    content = INDEX_HTML.read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "html.parser")

    missing = []
    for tag in soup.find_all(True):
        for attr in ("href", "src"):
            value = tag.get(attr)
            if not value or is_allowed_external(value) or value.startswith("#"):
                continue
            if value.startswith(("http://", "https://", "//", "/")):
                continue
            if not (SITE_ROOT / value).exists():
                missing.append(f"<{tag.name} {attr}=\"{value}\">")

    assert missing == [], "Missing local asset(s):\n" + "\n".join(f"  - {m}" for m in missing)
