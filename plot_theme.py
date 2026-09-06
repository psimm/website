"""Shared Lets-Plot theme for the site."""

from lets_plot import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
)

COLORS_CUSTOM = (
    "#2563eb",
    "#0d9488",
    "#db2777",
    "#7c3aed",
    "#d97706",
    "#16a34a",
    "#ea580c",
    "#0891b2",
)
COLOR_CUSTOM = COLORS_CUSTOM[0]
PLOT_BACKGROUND = "#ffffff"
PLOT_TEXT = "#0f172a"
PLOT_MUTED = "#64748b"
PLOT_RULE = "#cbd5e1"


def _typography(text: str, muted: str, title: str):
    return theme(
        text=element_text(family="Inter", size=11, color=text),
        plot_title=element_text(
            family="Inter", face="bold", size=18, color=title, hjust=0
        ),
        plot_subtitle=element_text(
            family="Inter", size=12, color=muted, hjust=0
        ),
        plot_caption=element_text(
            family="JetBrains Mono", size=9, color=muted, hjust=1
        ),
        axis_title=element_text(
            family="Inter", face="bold", size=10, color=text
        ),
        axis_text=element_text(family="Inter", size=10, color=muted),
        legend_title=element_text(
            family="JetBrains Mono", face="bold", size=9, color=muted
        ),
        legend_text=element_text(family="Inter", size=10, color=text),
        legend_position="bottom",
        legend_direction="horizontal",
        legend_box="vertical",
        legend_box_just="left",
        legend_spacing=8,
        legend_key_size=16,
        plot_margin=18,
    )


def theme_custom():
    return (
        theme_minimal()
        + _typography(PLOT_TEXT, PLOT_MUTED, PLOT_TEXT)
        + theme(
            plot_background=element_rect(
                fill=PLOT_BACKGROUND, color=PLOT_BACKGROUND
            ),
            panel_background=element_rect(
                fill=PLOT_BACKGROUND, color=PLOT_BACKGROUND
            ),
            panel_grid_major_x=element_blank(),
            panel_grid_minor=element_blank(),
            panel_grid_major_y=element_line(color="#e2e8f0", size=0.8),
            axis_ticks=element_blank(),
            axis_line=element_blank(),
            legend_background=element_rect(
                fill=PLOT_BACKGROUND, color=PLOT_BACKGROUND
            ),
            legend_key=element_rect(fill=PLOT_BACKGROUND, color=PLOT_BACKGROUND),
        )
    )


def scale_color_custom(**kwargs):
    return scale_color_manual(values=list(COLORS_CUSTOM), **kwargs)


def scale_fill_custom(**kwargs):
    return scale_fill_manual(values=list(COLORS_CUSTOM), **kwargs)
