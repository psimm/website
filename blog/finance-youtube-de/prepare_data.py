"""Prepare analysis-ready Parquet files for the finance YouTube article.

Run this script explicitly when the raw crawl or topic hierarchy changes. Quarto
only reads the generated files in ``data/``; it never calls GABRIEL itself.

uv run python blog/finance-youtube-de/prepare_data.py
uv run python blog/finance-youtube-de/prepare_data.py --limit 1000
uv run python blog/finance-youtube-de/prepare_data.py --bucket
"""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import gabriel
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

ARTICLE_DIR = Path(__file__).resolve().parent
RUN_DIR = ARTICLE_DIR / "runs" / "personal-finance-de"
RAW_DIR = RUN_DIR / "raw"
TOPIC_PATH = ARTICLE_DIR / "topic_descriptions.json"
DATA_DIR = ARTICLE_DIR / "data"
GABRIEL_DIR = ARTICLE_DIR / "gabriel"

MODEL = "gpt-5.6-luna"
EXPLORATION_VIDEO_LIMIT = 200
SAMPLE_SEED = 42
BUCKET_COUNT = 10
USE_FLEX_PROCESSING = True
BUCKET_PATH = ARTICLE_DIR / "bucket_definitions.md"
BUCKET_INSTRUCTIONS = (
    "These texts are German personal finance YouTube videos: title, description, "
    "and the first minute of the transcript. Propose mutually exclusive topic "
    "categories covering the main personal-finance themes. Use English names and "
    "definitions."
)

PARQUET_OPTIONS: dict[str, Any] = {
    "compression": "zstd",
    "compression_level": 9,
    "statistics": True,
    "row_group_size": 100_000,
}

BRAND_CHANNELS = (
    "HUK-COBURG",
    "ERGO Deutschland",
    "R+V Versicherung",
    "DVAG",
    "Mehr als Geld — by Sparkasse",
    "KT Bank AG",
)

EXCLUDED_CHANNELS = ("ARD Marktcheck",)

BROKER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Trade Republic", r"trade\s*republic"),
    ("Scalable Capital", r"scalable"),
    ("comdirect", r"comdirect"),
    ("Finanzen.net Zero", r"finanzen\.net\s*zero|finanzennet\s*zero"),
    ("Smartbroker", r"smartbroker"),
    ("Traders Place", r"traders?\s*place"),
    ("flatex", r"flatex"),
)
BROKER_FOCUS = ("Trade Republic", "Scalable Capital")


class TopicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SubTopic(TopicModel):
    pass


class Topic(TopicModel):
    subtopics: tuple[SubTopic, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def subtopic_names_are_unique(self) -> Topic:
        names = [subtopic.name.casefold() for subtopic in self.subtopics]
        if len(names) != len(set(names)):
            raise ValueError(f"Subtopic names must be unique within {self.name!r}")
        return self


class TopicHierarchy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topics: tuple[Topic, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def topic_names_are_unique(self) -> TopicHierarchy:
        names = [topic.name.casefold() for topic in self.topics]
        if len(names) != len(set(names)):
            raise ValueError("Top-level topic names must be unique")
        return self


def load_topic_hierarchy(path: Path = TOPIC_PATH) -> TopicHierarchy:
    return TopicHierarchy.model_validate_json(path.read_text(encoding="utf-8"))


def read_ndjson(filename: str) -> pl.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw input does not exist: {path}")
    return pl.read_ndjson(path, infer_schema_length=1_000_000)


def final_relevant_transcript_ids() -> pl.DataFrame:
    """Return video IDs whose latest crawler decision includes their transcript.

    Relevance decisions are append-only. A metadata-stage ``needs_transcript``
    record is provisional; only a final ``relevant`` decision at the transcript
    stage admits a downloaded transcript to this analysis.
    """
    path = RUN_DIR / "relevance_decision.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Relevance-decision input does not exist: {path}")
    decisions = pl.read_ndjson(path, infer_schema_length=1_000_000)
    required_columns = {"video_id", "label", "decision_point"}
    missing_columns = required_columns - set(decisions.columns)
    if missing_columns:
        raise ValueError(
            "relevance_decision.jsonl is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # JSONL order is the crawler's append order, so ``keep='last'`` gives the
    # final state even when an older run format has different audit columns.
    return (
        decisions.unique(subset="video_id", keep="last", maintain_order=True)
        .filter(
            (pl.col("label") == "relevant") & (pl.col("decision_point") == "transcript")
        )
        .select("video_id")
    )


def prepare_source_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Normalize raw SearchAPI video and transcript records."""
    raw_transcripts = read_ndjson("transcripts.jsonl")
    raw_videos = read_ndjson("video.jsonl")

    transcripts_nested = raw_transcripts.select(
        pl.col("search_parameters").struct.field("video_id"),
        pl.col("transcripts"),
    )
    transcripts = (
        transcripts_nested.explode("transcripts", empty_as_null=False)
        .unnest("transcripts")
        .select("video_id", "text", "start")
    )

    published_time_raw = pl.col("video").struct.field("published_time")
    published_time_text = published_time_raw.str.replace(r"^Premiered ", "")
    observed_at = (
        pl.col("search_metadata")
        .struct.field("created_at")
        .str.to_datetime("%Y-%m-%dT%H:%M:%SZ")
    )
    relative_hours = published_time_text.str.extract(r"^([0-9]+) hours? ago$", 1).cast(
        pl.Int64, strict=False
    )
    published_date = pl.coalesce(
        published_time_text.str.to_date("%b %d, %Y", strict=False),
        (observed_at - pl.duration(hours=relative_hours)).dt.date(),
    ).alias("published_time")

    unparsed_published_times = (
        raw_videos.select(
            published_time_raw.alias("published_time_raw"),
            published_date,
        )
        .filter(
            pl.col("published_time_raw").is_not_null()
            & pl.col("published_time").is_null()
        )
        .unique()
    )
    if not unparsed_published_times.is_empty():
        raise ValueError(
            f"Unsupported published_time values:\n{unparsed_published_times}"
        )

    transcript_video_ids = transcripts_nested.select("video_id").unique()
    videos = (
        raw_videos.select(
            pl.col("video").struct.field("id").alias("video_id"),
            pl.col("video").struct.field("title"),
            pl.col("video").struct.field("description"),
            pl.col("channel").struct.field("id").alias("channel_id"),
            pl.col("channel").struct.field("name").alias("channel_name"),
            pl.col("channel").struct.field("link").alias("channel_link"),
            pl.col("channel").struct.field("subscribers"),
            pl.col("video").struct.field("length_seconds"),
            pl.col("video").struct.field("views"),
            pl.col("video").struct.field("likes"),
            published_date,
        )
        .sort(["published_time", "video_id"], descending=[True, False])
        .join(
            transcript_video_ids.with_columns(pl.lit(True).alias("has_transcript")),
            how="left",
            on="video_id",
        )
        .with_columns(pl.col("has_transcript").fill_null(False))
    )

    relevant_transcript_ids = final_relevant_transcript_ids()
    videos = (
        videos.filter(pl.col("has_transcript"))
        .join(relevant_transcript_ids, on="video_id", how="inner")
        .filter(~pl.col("channel_name").is_in(EXCLUDED_CHANNELS))
    )
    transcripts = transcripts.join(
        videos.select("video_id"), on="video_id", how="inner"
    )
    return videos, transcripts


def build_top_channels_h_index(videos: pl.DataFrame) -> pl.DataFrame:
    organic = videos.with_columns(
        (
            pl.col("likes").fill_null(0)
            / pl.when(pl.col("views") > 0).then(pl.col("views"))
        ).alias("like_rate")
    ).filter(
        (pl.col("length_seconds") >= 180)
        & (pl.col("like_rate").fill_null(0) >= 0.005)
        & (~pl.col("channel_name").is_in(BRAND_CHANNELS))
    )
    h_index = (
        organic.sort(["channel_name", "views"], descending=[False, True])
        .with_columns(pl.int_range(1, pl.len() + 1).over("channel_name").alias("k"))
        .filter(pl.col("views") >= pl.col("k") * 10_000)
        .group_by("channel_name")
        .agg(pl.max("k").cast(pl.UInt16).alias("h"))
    )
    top_video = (
        organic.sort(
            ["channel_name", "views", "video_id"], descending=[False, True, False]
        )
        .group_by("channel_name", maintain_order=True)
        .agg(
            pl.first("title").alias("top_video_title"),
            pl.first("video_id").alias("top_video_id"),
            pl.first("views").alias("top_video_views"),
        )
    )
    return (
        organic.group_by("channel_name")
        .agg(
            pl.sum("views").alias("views_sum"),
            (pl.col("likes").fill_null(0).sum() / pl.sum("views")).alias("like_rate"),
        )
        .join(h_index, on="channel_name")
        .join(top_video, on="channel_name")
        .join(
            videos.group_by("channel_name").agg(
                pl.min("published_time").alias("first_published"),
                pl.col("channel_id").drop_nulls().first().alias("channel_id"),
                pl.col("channel_link").drop_nulls().first().alias("channel_link"),
                pl.col("subscribers").drop_nulls().first().alias("subscribers"),
            ),
            on="channel_name",
        )
        .sort(["h", "views_sum"], descending=True)
        .head(10)
        .with_columns(
            pl.coalesce(
                pl.when(pl.col("channel_link").str.len_chars() > 0).then(
                    pl.col("channel_link").str.replace(r"^http://", "https://")
                ),
                pl.concat_str(
                    pl.lit("https://www.youtube.com/channel/"),
                    pl.col("channel_id"),
                ),
            ).alias("channel_link"),
            pl.when(pl.col("channel_name").str.contains("Finanzfluss"))
            .then(pl.lit("focus"))
            .otherwise(pl.lit("rest"))
            .alias("tone"),
        )
        .drop("channel_id")
        .with_columns(pl.col("tone").cast(pl.Enum(["rest", "focus"])))
    )


def build_top_organic_videos(videos: pl.DataFrame) -> pl.DataFrame:
    organic = videos.with_columns(
        (
            pl.col("likes").fill_null(0)
            / pl.when(pl.col("views") > 0).then(pl.col("views"))
        ).alias("like_rate")
    ).filter(
        (pl.col("length_seconds") >= 180)
        & (pl.col("like_rate").fill_null(0) >= 0.005)
        & (~pl.col("channel_name").is_in(BRAND_CHANNELS))
    )
    return (
        organic.sort("views", descending=True)
        .head(5)
        .with_row_index("rank", offset=1)
        .select(
            pl.col("rank").cast(pl.UInt8),
            "video_id",
            "channel_name",
            "title",
            "published_time",
            "views",
            "like_rate",
        )
    )


def transcript_first_minute(transcripts: pl.DataFrame) -> pl.DataFrame:
    return (
        transcripts.filter(pl.col("start") < 60)
        .sort(["video_id", "start"])
        .group_by("video_id", maintain_order=True)
        .agg(pl.col("text").str.join(" ").alias("transcript_first_60s"))
    )


def exploration_sample(
    videos: pl.DataFrame,
    transcripts: pl.DataFrame,
    *,
    limit: int = EXPLORATION_VIDEO_LIMIT,
) -> pl.DataFrame:
    first_minute = transcript_first_minute(transcripts)
    eligible = videos.join(first_minute, on="video_id", how="inner")
    n = eligible.height if limit <= 0 else min(limit, eligible.height)
    sample = eligible.sample(n=n, seed=SAMPLE_SEED, shuffle=True).with_columns(
        pl.concat_str(
            pl.col("title").fill_null(""),
            pl.col("description").fill_null(""),
            pl.col("transcript_first_60s").fill_null(""),
            separator="\n",
        ).alias("text")
    )
    if sample.is_empty():
        raise ValueError("No videos with transcript text are available")
    return sample


def write_bucket_definitions(
    buckets: pd.DataFrame, path: Path, *, n_videos: int
) -> None:
    lines = [
        "# Candidate topic buckets",
        "",
        f"Model: `{MODEL}`",
        f"Videos: {n_videos}",
        "Text: title, description, and the first 60 seconds of transcript.",
        "",
    ]
    if buckets.empty:
        lines.append("No buckets were returned.")
    else:
        for row in buckets.itertuples(index=False):
            lines.append(f"## {row.bucket}")
            lines.append("")
            lines.append(str(row.definition).strip())
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


async def classify_topics(
    videos: pl.DataFrame,
    transcripts: pl.DataFrame,
    topics: tuple[Topic, ...],
    *,
    reset_files: bool,
    limit: int = EXPLORATION_VIDEO_LIMIT,
) -> pl.DataFrame:
    labels = {topic.name: topic.description for topic in topics}
    to_label = exploration_sample(videos, transcripts, limit=limit)

    result: pd.DataFrame = await gabriel.classify(
        df=to_label.to_pandas(),
        column_name="text",
        model=MODEL,
        labels=labels,
        service_tier="flex" if USE_FLEX_PROCESSING else None,
        save_dir=str(GABRIEL_DIR / "topics"),
        reset_files=reset_files,
    )
    classified = (
        pl.from_pandas(result)
        .rename({"predicted_classes": "topics"})
        .with_columns(pl.col("topics").cast(pl.List(pl.String)))
    )
    validate_labels(classified, "topics", set(labels))
    return classified


async def classify_subtopics(
    classified_videos: pl.DataFrame,
    topics: tuple[Topic, ...],
    *,
    reset_files: bool,
) -> pl.DataFrame:
    classifications: list[pl.DataFrame] = []

    for topic in topics:
        topic_items = classified_videos.filter(
            pl.col("topics").list.contains(topic.name)
        )
        if topic_items.is_empty():
            continue

        labels = {subtopic.name: subtopic.description for subtopic in topic.subtopics}
        to_label = topic_items.select(
            "video_id",
            pl.concat_str(
                pl.col("title").fill_null(""),
                pl.col("description").fill_null(""),
                separator="\n",
            ).alias("text"),
        )
        result: pd.DataFrame = await gabriel.classify(
            df=to_label.to_pandas(),
            column_name="text",
            model=MODEL,
            labels=labels,
            service_tier="flex" if USE_FLEX_PROCESSING else None,
            save_dir=str(GABRIEL_DIR / "subtopics" / topic.name),
            reset_files=reset_files,
        )
        classification = (
            pl.from_pandas(result)
            .rename({"predicted_classes": "subtopics"})
            .with_columns(
                pl.lit(topic.name).alias("topic"),
                pl.col("subtopics").cast(pl.List(pl.String)),
            )
            .select("video_id", "topic", "subtopics")
        )
        validate_labels(classification, "subtopics", set(labels))
        classifications.append(classification)

    if not classifications:
        return pl.DataFrame(
            schema={
                "video_id": pl.String,
                "topic": pl.String,
                "subtopics": pl.List(pl.String),
            }
        )
    return pl.concat(classifications)


def validate_labels(frame: pl.DataFrame, column: str, allowed_labels: set[str]) -> None:
    observed = set(
        frame.select(column)
        .explode(column, empty_as_null=False)
        .get_column(column)
        .drop_nulls()
        .to_list()
    )
    unknown = observed - allowed_labels
    if unknown:
        raise ValueError(f"Unexpected labels in {column}: {sorted(unknown)}")


def quarter_reference(first_date: Any, last_date: Any) -> pl.DataFrame:
    if first_date is None or last_date is None:
        raise ValueError("Classified videos have no publication date range")
    last_quarter = (last_date.month - 1) // 3 + 1
    labels = [
        f"{year} Q{quarter}"
        for year in range(first_date.year, last_date.year + 1)
        for quarter in range(1, 5)
        if (year, quarter) <= (last_date.year, last_quarter)
    ]
    return pl.DataFrame(
        {
            "quarter": labels,
            "quarter_order": range(len(labels)),
        },
        schema={"quarter": pl.String, "quarter_order": pl.UInt16},
    )


def add_quarter(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.format(
            "{} Q{}",
            pl.col("published_time").dt.year(),
            pl.col("published_time").dt.quarter(),
        ).alias("quarter")
    )


def build_broker_mentions_by_year(videos: pl.DataFrame) -> pl.DataFrame:
    broker_names = [name for name, _ in BROKER_PATTERNS]
    blob = videos.with_columns(
        pl.col("published_time").dt.year().alias("year"),
        pl.concat_str(
            pl.col("title").fill_null(""),
            pl.lit(" "),
            pl.col("description").fill_null(""),
        )
        .str.to_lowercase()
        .alias("blob"),
    ).with_columns(
        [
            pl.col("blob").str.contains(pattern).alias(name)
            for name, pattern in BROKER_PATTERNS
        ]
    )
    return (
        blob.unpivot(
            index=["year"],
            on=broker_names,
            variable_name="broker",
            value_name="hit",
        )
        .group_by("year", "broker")
        .agg(
            pl.len().cast(pl.UInt32).alias("videos"),
            pl.col("hit").sum().cast(pl.UInt32).alias("hits"),
        )
        .with_columns(
            (pl.col("hits") / pl.col("videos") * 100).alias("share"),
            pl.when(pl.col("broker").is_in(list(BROKER_FOCUS)))
            .then(pl.lit("focus"))
            .otherwise(pl.lit("context"))
            .alias("tone"),
        )
        .with_columns(
            pl.col("year").cast(pl.UInt16),
            pl.col("broker").cast(pl.Enum(broker_names)),
            pl.col("tone").cast(pl.Enum(["context", "focus"])),
        )
        .select("year", "broker", "videos", "hits", "share", "tone")
        .sort(["broker", "year"])
    )


def build_videos_by_quarter(videos: pl.DataFrame) -> pl.DataFrame:
    first_published = videos.get_column("published_time").min()
    last_published = videos.get_column("published_time").max()
    quarters = quarter_reference(first_published, last_published)
    quarter_enum = pl.Enum(quarters.get_column("quarter").to_list())
    return (
        quarters.join(
            add_quarter(videos)
            .group_by("quarter")
            .agg(pl.len().cast(pl.UInt32).alias("videos")),
            on="quarter",
            how="left",
        )
        .with_columns(
            pl.col("videos").fill_null(0),
            pl.col("quarter").cast(quarter_enum),
        )
        .sort("quarter_order")
    )


def build_render_tables(
    videos: pl.DataFrame,
    topics: tuple[Topic, ...],
    classified_videos: pl.DataFrame,
    classified_subtopics: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Build compact, plot-ready tables with ordering encoded in the schema."""
    canonical_videos = classified_videos.select(
        "video_id",
        "title",
        "channel_name",
        "published_time",
        "views",
        "topics",
    ).sort(["published_time", "video_id"], descending=[True, False])

    topic_video_rows = (
        canonical_videos.select("video_id", "channel_name", "published_time", "topics")
        .explode("topics", empty_as_null=False)
        .rename({"topics": "topic"})
        .filter(pl.col("topic").is_not_null())
    )
    topic_names = [topic.name for topic in topics]
    topic_counts = (
        pl.DataFrame({"topic": topic_names})
        .join(
            topic_video_rows.group_by("topic").agg(
                pl.len().cast(pl.UInt32).alias("videos")
            ),
            on="topic",
            how="left",
        )
        .with_columns(pl.col("videos").fill_null(0))
        .sort(["videos", "topic"], descending=[True, False])
        .with_row_index("topic_order")
        .with_columns(pl.col("topic_order").cast(pl.UInt16))
    )
    ordered_topics = topic_counts.get_column("topic").to_list()
    topic_enum = pl.Enum(ordered_topics)

    channels = videos.group_by("channel_name").agg(
        pl.len().cast(pl.UInt32).alias("videos"),
        pl.mean("views").round().cast(pl.Int64).alias("views_mean"),
    )
    top_channels_h_index = build_top_channels_h_index(videos)
    top_organic_videos = build_top_organic_videos(videos)

    top_channels = (
        topic_video_rows.group_by("topic", "channel_name")
        .agg(pl.len().cast(pl.UInt32).alias("videos"))
        .sort(
            ["topic", "videos", "channel_name"],
            descending=[False, True, False],
        )
        .group_by("topic", maintain_order=True)
        .head(8)
        .join(topic_counts.select("topic", "topic_order"), on="topic")
        .with_columns(pl.col("topic").cast(topic_enum))
        .sort(
            ["topic_order", "videos", "channel_name"], descending=[False, True, False]
        )
    )

    first_published = canonical_videos.get_column("published_time").min()
    last_published = canonical_videos.get_column("published_time").max()
    quarters = quarter_reference(first_published, last_published)
    quarter_labels = quarters.get_column("quarter").to_list()
    quarter_enum = pl.Enum(quarter_labels)

    topic_quarter_hits = (
        add_quarter(topic_video_rows)
        .group_by("topic", "quarter")
        .agg(pl.len().cast(pl.UInt32).alias("videos"))
    )
    topic_quarter = (
        topic_counts.select("topic", "topic_order")
        .join(quarters, how="cross")
        .join(topic_quarter_hits, on=["topic", "quarter"], how="left")
        .with_columns(
            pl.col("videos").fill_null(0),
            pl.col("topic").cast(topic_enum),
            pl.col("quarter").cast(quarter_enum),
        )
        .sort(["topic_order", "quarter_order"])
    )

    subtopic_reference = pl.DataFrame(
        [
            {
                "topic": topic.name,
                "subtopic": subtopic.name,
                "subtopic_order": subtopic_order,
            }
            for topic in topics
            for subtopic_order, subtopic in enumerate(topic.subtopics)
        ],
        schema={
            "topic": pl.String,
            "subtopic": pl.String,
            "subtopic_order": pl.UInt16,
        },
    )
    subtopic_hits = (
        classified_subtopics.explode("subtopics", empty_as_null=True)
        .rename({"subtopics": "subtopic"})
        .filter(pl.col("subtopic").is_not_null())
        .group_by("topic", "subtopic")
        .agg(pl.len().cast(pl.UInt32).alias("subtopic_videos"))
    )
    subtopic_counts = (
        subtopic_reference.join(subtopic_hits, on=["topic", "subtopic"], how="left")
        .with_columns(pl.col("subtopic_videos").fill_null(0))
        .join(
            topic_counts.select("topic", "videos", "topic_order").rename(
                {"videos": "topic_videos"}
            ),
            on="topic",
        )
        .with_columns(
            pl.concat_str(
                pl.format("{} ({} video", "topic", "topic_videos"),
                pl.when(pl.col("topic_videos") == 1)
                .then(pl.lit(""))
                .otherwise(pl.lit("s")),
                pl.lit(")"),
            ).alias("topic_facet")
        )
        .sort(
            ["topic_order", "subtopic_videos", "subtopic_order"],
            descending=[False, True, False],
        )
    )
    topic_facets = (
        subtopic_counts.select("topic_order", "topic_facet")
        .unique()
        .sort("topic_order")
        .get_column("topic_facet")
        .to_list()
    )
    subtopic_counts = subtopic_counts.with_columns(
        pl.col("topic").cast(topic_enum),
        pl.col("topic_facet").cast(pl.Enum(topic_facets)),
    )

    subtopic_quarter_hits = (
        classified_subtopics.join(
            canonical_videos.select("video_id", "published_time"),
            on="video_id",
            how="left",
        )
        .explode("subtopics", empty_as_null=False)
        .rename({"subtopics": "subtopic"})
        .filter(pl.col("subtopic").is_not_null())
        .pipe(add_quarter)
        .group_by("topic", "subtopic", "quarter")
        .agg(pl.len().cast(pl.UInt32).alias("videos"))
    )
    all_subtopics = list(
        dict.fromkeys(subtopic.name for topic in topics for subtopic in topic.subtopics)
    )
    subtopic_quarter = (
        subtopic_reference.join(quarters, how="cross")
        .join(
            subtopic_quarter_hits,
            on=["topic", "subtopic", "quarter"],
            how="left",
        )
        .join(topic_counts.select("topic", "topic_order"), on="topic")
        .with_columns(
            pl.col("videos").fill_null(0),
            pl.col("topic").cast(topic_enum),
            pl.col("subtopic").cast(pl.Enum(all_subtopics)),
            pl.col("quarter").cast(quarter_enum),
        )
        .sort(["topic_order", "subtopic_order", "quarter_order"])
    )

    summary = pl.DataFrame(
        {
            "videos_with_transcript": [videos.height],
            "start_date": [videos.get_column("published_time").min()],
            "end_date": [videos.get_column("published_time").max()],
        },
        schema={
            "videos_with_transcript": pl.UInt32,
            "start_date": pl.Date,
            "end_date": pl.Date,
        },
    )

    return {
        "summary": summary,
        "channels": channels,
        "top_channels_h_index": top_channels_h_index,
        "top_organic_videos": top_organic_videos,
        "classified_videos": canonical_videos,
        "classified_subtopics": classified_subtopics.sort(["topic", "video_id"]),
        "topic_counts": topic_counts.with_columns(pl.col("topic").cast(topic_enum)),
        "top_channels_by_topic": top_channels,
        "videos_by_quarter": build_videos_by_quarter(videos),
        "broker_mentions_by_year": build_broker_mentions_by_year(videos),
        "topic_popularity_by_quarter": topic_quarter,
        "subtopic_popularity_by_quarter": subtopic_quarter,
        "subtopic_counts": subtopic_counts,
    }


def write_tables(tables: dict[str, pl.DataFrame]) -> None:
    """Write a complete set via temporary files to avoid partial updates."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="finance-youtube-", dir=DATA_DIR.parent
    ) as tmp:
        temporary_dir = Path(tmp)
        for name, frame in tables.items():
            frame.write_parquet(temporary_dir / f"{name}.parquet", **PARQUET_OPTIONS)
        for name in tables:
            os.replace(
                temporary_dir / f"{name}.parquet",
                DATA_DIR / f"{name}.parquet",
            )


async def run_bucket(
    *, reset_gabriel: bool, limit: int = EXPLORATION_VIDEO_LIMIT
) -> Path:
    videos, transcripts = prepare_source_data()
    sample = exploration_sample(videos, transcripts, limit=limit)
    result: pd.DataFrame = await gabriel.bucket(
        df=sample.to_pandas(),
        column_name="text",
        model=MODEL,
        bucket_count=BUCKET_COUNT,
        additional_instructions=BUCKET_INSTRUCTIONS,
        service_tier="flex" if USE_FLEX_PROCESSING else None,
        save_dir=str(GABRIEL_DIR / "buckets"),
        reset_files=reset_gabriel,
    )
    write_bucket_definitions(result, BUCKET_PATH, n_videos=sample.height)
    return BUCKET_PATH


def load_classified_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    videos_path = DATA_DIR / "classified_videos.parquet"
    subtopics_path = DATA_DIR / "classified_subtopics.parquet"
    missing = [str(path) for path in (videos_path, subtopics_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot reuse classified tables. Missing: " + ", ".join(missing)
        )
    return pl.read_parquet(videos_path), pl.read_parquet(subtopics_path)


async def run(
    *,
    reset_gabriel: bool,
    reuse_classified: bool = False,
    limit: int = EXPLORATION_VIDEO_LIMIT,
) -> dict[str, pl.DataFrame]:
    hierarchy = load_topic_hierarchy()
    videos, transcripts = prepare_source_data()
    if reuse_classified:
        classified_videos, classified_subtopics = load_classified_tables()
        keep = videos.select("video_id")
        classified_videos = classified_videos.join(keep, on="video_id", how="inner")
        classified_subtopics = classified_subtopics.join(
            keep, on="video_id", how="inner"
        )
    else:
        classified_videos = await classify_topics(
            videos,
            transcripts,
            hierarchy.topics,
            reset_files=reset_gabriel,
            limit=limit,
        )
        classified_subtopics = await classify_subtopics(
            classified_videos,
            hierarchy.topics,
            reset_files=reset_gabriel,
        )
    return build_render_tables(
        videos,
        hierarchy.topics,
        classified_videos,
        classified_subtopics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Parquet inputs for index.qmd. This performs GABRIEL API "
            "classification."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=EXPLORATION_VIDEO_LIMIT,
        metavar="N",
        help=(
            "Maximum videos to classify or bucket "
            f"(default: {EXPLORATION_VIDEO_LIMIT}). 0 means no limit."
        ),
    )
    parser.add_argument(
        "--reset-gabriel",
        action="store_true",
        help="Discard compatible GABRIEL checkpoints and run from scratch.",
    )
    parser.add_argument(
        "--bucket",
        action="store_true",
        help=(
            "Discover candidate topic buckets with gabriel.bucket on the same "
            "sample as classification, write a markdown file, and exit."
        ),
    )
    parser.add_argument(
        "--reuse-classified",
        action="store_true",
        help=(
            "Rebuild Parquet tables from existing classified_*.parquet files "
            "without calling GABRIEL."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    load_dotenv(ARTICLE_DIR / ".env")
    load_dotenv(ARTICLE_DIR.parents[1] / ".env")
    if args.bucket:
        path = asyncio.run(
            run_bucket(reset_gabriel=args.reset_gabriel, limit=args.limit)
        )
        print(f"Wrote bucket definitions to {path}")
        return
    tables = asyncio.run(
        run(
            reset_gabriel=args.reset_gabriel,
            reuse_classified=args.reuse_classified,
            limit=args.limit,
        )
    )
    write_tables(tables)
    print(f"Wrote {len(tables)} Parquet files to {DATA_DIR}")
    for name, frame in tables.items():
        print(f"  {name}.parquet: {frame.height:,} rows x {frame.width} columns")


if __name__ == "__main__":
    main()
