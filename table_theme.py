"""Shared Great Tables theme for the site."""

from great_tables import GT, loc, style

TABLE_BACKGROUND = "#ffffff"
TABLE_TEXT = "#0f172a"
TABLE_MUTED = "#64748b"
TABLE_RULE = "#e2e8f0"
TABLE_RULE_STRONG = "#cbd5e1"


def theme_custom(table: GT) -> GT:
    """Apply the site's light, horizontal-rule table theme to a GT table."""
    return (
        table.opt_row_striping(row_striping=False)
        .tab_options(
            container_width="100%",
            table_width="100%",
            table_margin_left="auto",
            table_margin_right="auto",
            table_background_color=TABLE_BACKGROUND,
            table_font_names=[
                "Inter",
                "-apple-system",
                "BlinkMacSystemFont",
                "sans-serif",
            ],
            table_font_size="15px",
            table_font_color=TABLE_TEXT,
            table_border_top_style="none",
            table_border_bottom_style="solid",
            table_border_bottom_width="1px",
            table_border_bottom_color=TABLE_RULE_STRONG,
            table_border_left_style="none",
            table_border_right_style="none",
            heading_background_color=TABLE_BACKGROUND,
            heading_align="left",
            heading_title_font_size="18px",
            heading_title_font_weight="bold",
            heading_subtitle_font_size="12px",
            heading_subtitle_font_weight="normal",
            heading_padding="4px",
            heading_padding_horizontal="8px",
            heading_border_bottom_style="none",
            heading_border_lr_style="none",
            column_labels_background_color=TABLE_BACKGROUND,
            column_labels_font_size="14px",
            column_labels_font_weight="bold",
            column_labels_padding="7px",
            column_labels_padding_horizontal="8px",
            column_labels_vlines_style="none",
            column_labels_border_top_style="solid",
            column_labels_border_top_width="1px",
            column_labels_border_top_color=TABLE_RULE_STRONG,
            column_labels_border_bottom_style="solid",
            column_labels_border_bottom_width="1px",
            column_labels_border_bottom_color=TABLE_RULE_STRONG,
            column_labels_border_lr_style="none",
            table_body_hlines_style="solid",
            table_body_hlines_width="1px",
            table_body_hlines_color=TABLE_RULE,
            table_body_vlines_style="none",
            table_body_border_top_style="none",
            table_body_border_bottom_style="solid",
            table_body_border_bottom_width="1px",
            table_body_border_bottom_color=TABLE_RULE_STRONG,
            stub_background_color=TABLE_BACKGROUND,
            stub_border_style="none",
            stub_row_group_border_style="none",
            data_row_padding="7px",
            data_row_padding_horizontal="8px",
            source_notes_background_color=TABLE_BACKGROUND,
            source_notes_border_lr_style="none",
            row_striping_background_color=TABLE_BACKGROUND,
            row_striping_include_stub=False,
            row_striping_include_table_body=False,
            quarto_disable_processing=True,
        )
        .tab_style(style=style.text(color=TABLE_MUTED), locations=loc.subtitle())
        .tab_style(
            style=style.text(
                color=TABLE_MUTED,
                font="JetBrains Mono",
                size="12px",
            ),
            locations=loc.source_notes(),
        )
    )
