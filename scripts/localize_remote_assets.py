#!/usr/bin/env python3
"""Replace remotely hosted frontend libraries with vendored site assets."""

from __future__ import annotations

import re
from pathlib import Path


OUTPUT_DIR = Path("_site")

REPLACEMENTS = {
    "//gc.zgo.at/count.js": "/assets/vendor/goatcounter/count.js",
    "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js": (
        "/assets/vendor/jquery/jquery-3.5.1.min.js"
    ),
    "https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.min.js": (
        "/assets/vendor/jquery/jquery-3.5.1.min.js"
    ),
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js": (
        "/assets/vendor/requirejs/require-2.3.6.min.js"
    ),
    "https://cdn.jsdelivr.net/npm/requirejs@2.3.6/require.min.js": (
        "/assets/vendor/requirejs/require-2.3.6.min.js"
    ),
    "https://cdn.plot.ly/plotly-3.3.0.min.js": (
        "/assets/vendor/plotly/plotly-3.3.0.min.js"
    ),
    "https://cdn.plot.ly/plotly-3.3.0.min": (
        "/assets/vendor/plotly/plotly-3.3.0.min.js"
    ),
    "https://cdn.plot.ly/plotly-2.26.0.min": (
        "/assets/vendor/plotly/plotly-2.26.0.min"
    ),
    "https://cdn.datatables.net/1.12.1/js/jquery.dataTables.mjs": (
        "/assets/vendor/datatables/jquery.dataTables-1.12.1.mjs"
    ),
    "https://cdn.datatables.net/1.13.1/css/jquery.dataTables.min.css": (
        "/assets/vendor/datatables/jquery.dataTables-1.13.1.min.css"
    ),
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js": (
        "/assets/vendor/mathjax/tex-svg-full-3.2.2.js"
    ),
}

# Modern browsers do not need Quarto's legacy ES6 polyfill. Plotly inserts
# MathJax 2 even for figures without TeX labels; the site-wide local MathJax 3
# bundle handles article equations, so the redundant remote loader is removed.
REMOVED_SCRIPT_HOSTS = (
    "cdnjs.cloudflare.com/polyfill/",
    "cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js",
)

REMOTE_SCRIPT = re.compile(r'<script\b[^>]*\bsrc=["\'](?:https?:)?//[^"\']+["\'][^>]*>')
REMOTE_STYLESHEET = re.compile(
    r'<link\b(?=[^>]*\brel=["\']stylesheet["\'])[^>]*'
    r'\bhref=["\'](?:https?:)?//[^"\']+["\'][^>]*>'
)

REMOTE_LIBRARY_HOSTS = (
    "cdnjs.cloudflare.com",
    "cdn.jsdelivr.net",
    "cdn.plot.ly",
    "cdn.datatables.net",
    "gc.zgo.at",
)


def localize(html: str) -> str:
    for remote, local in REPLACEMENTS.items():
        html = html.replace(remote, local)

    for host_fragment in REMOVED_SCRIPT_HOSTS:
        html = re.sub(
            rf'<script\b[^>]*\bsrc=["\']https://{re.escape(host_fragment)}'
            rf'[^"\']*["\'][^>]*></script>',
            "",
            html,
        )

    return html


def main() -> None:
    if not OUTPUT_DIR.is_dir():
        raise SystemExit(f"Output directory does not exist: {OUTPUT_DIR}")

    failures: list[str] = []
    for path in OUTPUT_DIR.rglob("*.html"):
        original = path.read_text(encoding="utf-8")
        localized = localize(original)
        if localized != original:
            path.write_text(localized, encoding="utf-8")

        remote_tags = REMOTE_SCRIPT.findall(localized) + REMOTE_STYLESHEET.findall(localized)
        if remote_tags:
            failures.append(f"{path}: {' '.join(remote_tags)}")

        remaining_hosts = [host for host in REMOTE_LIBRARY_HOSTS if host in localized]
        if remaining_hosts:
            failures.append(f"{path}: remote library hosts remain: {', '.join(remaining_hosts)}")

    if failures:
        details = "\n".join(failures)
        raise SystemExit(f"Remote frontend dependencies remain:\n{details}")


if __name__ == "__main__":
    main()
