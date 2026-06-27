#!/usr/bin/env python3
"""Build token-viz.html and token-viz.png for the save-token-costs article."""

from __future__ import annotations

import html
from pathlib import Path

import tiktoken
from playwright.sync_api import sync_playwright

DIR = Path(__file__).resolve().parent
HTML_PATH = DIR / "token-viz.html"
PNG_PATH = DIR / "token-viz.png"

# Accurate translation: "Please summarize this article in three bullet points."
EN_TEXT = "Please summarize this article in three bullet points."
JA_TEXT = "この記事を3つの箇条書きで要約してください。"

# Semantic groups (shared color = same phrase in both languages).
# EN token index -> group id
EN_GROUP = {
    0: "please",       # Please
    1: "summarize",    #  summarize
    2: "this_article", #  this
    3: "this_article", #  article
    4: "in",           #  in
    5: "three",        #  three
    6: "bullet_points", #  bullet
    7: "bullet_points", #  points
    8: "period",       # .
}

# JA token index -> group id (order from o200k_base)
JA_GROUP = {
    0: "this_article",   # この記事
    1: "this_article",   # を
    2: "three",          # 3
    3: "three",          # つ
    4: "three",          # の
    5: "bullet_points",  # (byte fragment of 箇)
    6: "bullet_points",  # 箇
    7: "bullet_points",  # 条
    8: "bullet_points",  # 書
    9: "bullet_points",  # き
    10: "in",            # で
    11: "summarize",     # 要
    12: "summarize",     # 約
    13: "please",        # してください
    14: "period",        # 。
}

GROUP_COLORS = {
    "this_article": ("#2563eb", "#dbeafe"),
    "three": ("#7c3aed", "#ede9fe"),
    "bullet_points": ("#059669", "#d1fae5"),
    "in": ("#d97706", "#fef3c7"),
    "summarize": ("#db2777", "#fce7f3"),
    "please": ("#0891b2", "#cffafe"),
    "period": ("#dc2626", "#fee2e2"),
}


def token_slices(enc: tiktoken.Encoding, text: str) -> list[str]:
    """Map each token to the substring it adds in the original text."""
    data = text.encode("utf-8")
    offset = 0
    decoded_so_far = ""
    slices: list[str] = []
    for token in enc.encode(text):
        end = offset + len(enc.decode_single_token_bytes(token))
        chunk = data[:end]
        j = len(chunk)
        decoded = None
        while j > 0:
            try:
                decoded = chunk[:j].decode("utf-8")
                break
            except UnicodeDecodeError:
                j -= 1
        if decoded is None:
            piece = ""
        else:
            piece = decoded[len(decoded_so_far) :]
            decoded_so_far = decoded
        slices.append(piece)
        offset = end
    return slices


def chunks_html(
    enc: tiktoken.Encoding,
    text: str,
    group_by_index: dict[int, str],
) -> tuple[int, str]:
    slices = token_slices(enc, text)
    parts: list[str] = []
    for i, piece in enumerate(slices):
        group = group_by_index[i]
        border, bg = GROUP_COLORS[group]
        display = html.escape(piece) if piece else "·"
        partial = not piece
        cls = "token-chunk token-chunk-partial" if partial else "token-chunk"
        parts.append(
            f'<span class="{cls}" style="--tb:{border};--tg:{bg}" title="token {i}">'
            f'<span class="token-idx">{i}</span>{display}</span>'
        )
    return len(slices), "".join(parts)


def build_html() -> str:
    enc = tiktoken.get_encoding("o200k_base")
    en_n, en_html = chunks_html(enc, EN_TEXT, EN_GROUP)
    ja_n, ja_html = chunks_html(enc, JA_TEXT, JA_GROUP)
    pct = round((ja_n / en_n - 1) * 100)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Token boundary comparison</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
      background: #f8fafc;
      color: #0f172a;
      padding: 24px;
      width: 920px;
    }}
    .token-viz {{
      border: 1px solid #cbd5e1;
      background: #f8fafc;
    }}
    .token-viz-head {{
      padding: 0.65rem 1rem;
      border-bottom: 1px solid #cbd5e1;
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #64748b;
    }}
    .token-viz-row {{
      padding: 1rem 1rem 0.25rem;
      border-bottom: 1px solid #e2e8f0;
    }}
    .token-viz-row:last-of-type {{ border-bottom: none; }}
    .token-viz-meta {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 0.65rem;
    }}
    .token-viz-lang {{ font-weight: 600; font-size: 0.95rem; }}
    .token-viz-count {{
      font-family: "JetBrains Mono", Consolas, monospace;
      font-size: 0.82rem;
      color: #475569;
    }}
    .token-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      line-height: 1.55;
      font-size: 1.05rem;
    }}
    .token-chunk {{
      display: inline-flex;
      align-items: baseline;
      gap: 0.25rem;
      padding: 0.2rem 0.45rem 0.28rem;
      border: 2px solid var(--tb);
      background: var(--tg);
      font-family: "JetBrains Mono", Consolas, monospace;
      font-size: 0.92rem;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .token-chunk-partial {{
      border-style: dashed;
      opacity: 0.85;
      color: var(--tb);
    }}
    .token-idx {{
      font-size: 0.62rem;
      line-height: 1;
      color: var(--tb);
      opacity: 0.85;
      min-width: 0.85rem;
      text-align: center;
    }}
    .token-viz-foot {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      padding: 0.75rem 1rem;
      border-top: 1px solid #cbd5e1;
      background: #fff;
      font-size: 0.88rem;
      color: #334155;
    }}
    .token-viz-ratio {{
      font-family: "JetBrains Mono", Consolas, monospace;
      font-weight: 600;
      color: #b45309;
    }}
    code {{
      font-family: "JetBrains Mono", Consolas, monospace;
      font-size: 0.85em;
    }}
  </style>
</head>
<body>
  <div class="token-viz">
    <div class="token-viz-head">
      Same sentence · <code>o200k_base</code> tokenizer · matching colors = same phrase
    </div>
    <div class="token-viz-row">
      <div class="token-viz-meta">
        <span class="token-viz-lang">English</span>
        <span class="token-viz-count"><strong>{en_n}</strong> tokens</span>
      </div>
      <div class="token-line">{en_html}</div>
    </div>
    <div class="token-viz-row">
      <div class="token-viz-meta">
        <span class="token-viz-lang">Japanese</span>
        <span class="token-viz-count"><strong>{ja_n}</strong> tokens</span>
      </div>
      <div class="token-line">{ja_html}</div>
    </div>
    <div class="token-viz-foot">
      <span>Each box is one billed token.</span>
      <span class="token-viz-ratio">+{pct}% tokens in Japanese</span>
    </div>
  </div>
</body>
</html>
"""


def screenshot(html_path: Path, png_path: Path) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 920, "height": 480})
        page.goto(html_path.as_uri())
        page.wait_for_timeout(500)
        viz = page.locator(".token-viz")
        viz.screenshot(path=str(png_path))
        browser.close()


def main() -> None:
    HTML_PATH.write_text(build_html(), encoding="utf-8")
    screenshot(HTML_PATH, PNG_PATH)
    print(f"Wrote {HTML_PATH.name} and {PNG_PATH.name}")


if __name__ == "__main__":
    main()
