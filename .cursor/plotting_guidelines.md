# Plotting guidelines

## Library

**Lets-Plot is the plotting library of choice.** Use it for all new charts and
keep one article on one consistent visual system.

## Theme and color

Use the shared theme rather than copying its values into an article:

```python
sys.path.insert(0, str(Path.cwd().parents[1]))
from plot_theme import COLOR_CUSTOM, theme_custom
```

Add `theme_custom()` to every chart. Use `COLOR_CUSTOM` for a single highlighted
series. For categorical mappings, import and add `scale_color_custom()` or
`scale_fill_custom()`.

Gray is for context: reference lines, uncertainty, historical values, disabled
categories, or observations that should recede. Primary data should normally
use color. Do not rely on color alone when a distinction is essential.

## Typography

- Use **Inter** throughout the chart. It's already configured in the theme.
- Reserve **JetBrains Mono** for captions and occasional technical metadata.
- Keep titles concise, left aligned, and semibold or bold.
- Use subtitles for the measure, comparison, and time window.
- Use captions for sources, denominators, and methodological context.

## Quarto

Keep plot output within the article column:

```css
.article-plot .cell-output-display,
.article-plot .cell-output-display > div,
.article-plot svg,
.article-plot iframe { max-width: 100%; }

.article-plot .cell-output-display { overflow-x: clip; }
```

Initialize Lets-Plot once in a hidden-code setup cell, but retain its output so
the JavaScript runtime remains available:

```python
LetsPlot.setup_html(
    isolated_frame=False,
    offline=True,
    responsive=True,
    width_mode="scaled",
    height_mode="scaled",
    width=680,
    height=480,
    force_immediate_render=False,
)
```

Use `ggsize(width, height)` for each chart. Keep the width at or below 680 px
and adjust height to the content. Horizontal scrollbars are a defect; fix the
canvas, labels, facets, or container rather than requiring sideways scrolling.

## Axes, grids, and layout

- Use either horizontal or vertical grids. Only use both if absolutely needed.
- Don't use minor grids.
- If grids are absent, retain ticks or another visual anchor.
- Use heatmap tile boundaries or panel grids, but not both.
- Format numbers and dates for readers; avoid redundant years and long decimals.
- Remove axis titles that repeat obvious information. Use `x=""` or `y=""`.
- Put legends below the plot and stack different guides vertically.
- Use short legend headings and keep categorical guides before size guides.

## Ordering and tooltips

Encode ordering explicitly with `as_discrete()`:

```python
aes(x=as_discrete("category", order_by="value", order=1), y="value")
```

Tooltips should add exact values that are difficult to read from the chart.
Give each field a clear label and an explicit date, number, percentage, or
currency format. Do not repeat labels that are already printed clearly.

## Content

- State the finding in the title when the analysis supports one.
- Preserve meaningful zero observations, e.g. a time series having an implicit zero value on a day. This may require joining the data frame with a reference zero observations data frame.
- State aggregation periods, denominators, and multi-label counting rules.
- Keep caveats and sampling limits visible.
- Interactive details should complement a chart whose main point remains clear
  without interaction.

## Publication checklist

1. Render every chart and check for renderer errors.
2. Inspect typography, ordering, legends, grids, clipping, and empty space.
3. Check a narrow viewport and confirm there is no horizontal scrollbar.
4. Confirm the page contains one runtime initialization and responsive output.
5. Ensure rendering uses prepared local data and triggers no external analysis.
