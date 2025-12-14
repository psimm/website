# Website

Quarto website with per-post Python environments.

## Rendering

**Full site (uses frozen/cached Python outputs):**
```bash
quarto render
```

**Individual post with specific dependencies:**
```bash
cd blog/postname
uv sync
uv run quarto render index.qmd
```

## Per-Post Dependencies

Posts needing specific package versions have their own `pyproject.toml` and `uv.lock` in their directory. When rendered with `uv run`, they use their isolated environment.

The `_freeze/` directory caches Python execution results. Full site renders use these cached outputs without re-executing Python.

## Adding a New Post with Specific Dependencies

1. Create `blog/newpost/pyproject.toml` with required packages
2. Run `cd blog/newpost && uv sync && uv run quarto render index.qmd`
3. Commit the `_freeze/` output

