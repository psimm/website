#!/usr/bin/env python3
"""Generate author-footer stub and shared HTML fragment from YAML."""

from __future__ import annotations

import html
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "includes" / "author-footer.yml"
STUB_PATH = ROOT / "includes" / "_author-footer.stub.html"
FRAGMENT_PATH = ROOT / "assets" / "author-footer.html"


def md_links_to_html(text: str) -> str:
    """Escape text and convert [label](url) markdown links to HTML anchors."""
    escaped = html.escape(re.sub(r"\s+", " ", text.strip()))
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)


def write_if_changed(path: Path, content: str) -> bool:
    """Write content only when it differs, so Quarto does not see a dirty mtime."""
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    name = html.escape(config["name"])
    image = html.escape(config["image"])
    image_alt = html.escape(config.get("image-alt") or config["name"])
    label = html.escape(config.get("label") or "About the author")
    bio = md_links_to_html(config["bio"])
    more_text = html.escape(config.get("more-text") or "Read more →")
    more_href = html.escape(config.get("more-href") or "/about.html")

    stub = """\
<div id="author-footer"></div>
<script src="/assets/js/author-footer.js"></script>
"""

    fragment = f"""\
<nav class="blog-post-nav" id="blog-post-nav" aria-label="Post navigation" hidden>
  <div class="blog-post-nav-prev" id="blog-post-nav-prev" hidden>
    <div class="blog-post-nav-label">← Previous</div>
    <a class="blog-post-nav-link" href="#"></a>
  </div>
  <div class="blog-post-nav-next" id="blog-post-nav-next" hidden>
    <div class="blog-post-nav-label">Next →</div>
    <a class="blog-post-nav-link" href="#"></a>
  </div>
</nav>

<section class="about-the-author" aria-label="{label}">
  <img
    class="about-the-author-photo"
    src="{image}"
    alt="{image_alt}"
    width="140"
    height="140"
    loading="lazy"
  >
  <div class="about-the-author-label">{label}</div>
  <h3 class="about-the-author-name">{name}</h3>
  <p class="about-the-author-bio">
    {bio}
    <a class="about-the-author-more" href="{more_href}">{more_text}</a>
  </p>
</section>
"""

    for path, content in ((STUB_PATH, stub), (FRAGMENT_PATH, fragment)):
        if write_if_changed(path, content):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
