---
title: "Data frame wars: pandas vs. polars vs duckdb"
excerpt: "A comparison of pandas, polars and duckdb from the perspective of a dplyr user."
slug: "dataframes"
author: "Paul Simmering"
date: "2021-12-20"
categories: ["R", "Python"]
tags: ["Performance"]
---



I'm a long time R user and lately I've seen more and more signals that it's worth investing into learning more Python. I use it for NLP with [spaCy](https://spacy.io) and to build functions on [AWS Lambda](https://aws.amazon.com/lambda/features/) (though the recent [lambdr](https://mdneuzerling.com/post/serverless-on-demand-parametrised-r-markdown-reports-with-aws-lambda/) package gave R some great tools there too). Further, there are many more data API libraries available for Python than for R.

## The contenders

The first thing I needed was a way to manipulate data frames. I've narrowed it down to 3 choices:

- [Pandas](https://pandas.pydata.org): The most commonly used library and the one with the most tutorials and Stack Overflow answers available.
- [Polars](https://www.pola.rs): The fastest library available. It's a new library that doesn't have nearly as many users or help available. But according to the [H2Oai ops benchmark](https://h2oai.github.io/db-benchmark/), it often runs 10x faster than Pandas.
- [Duckdb](https://www.pola.rs): Use an in-memory OLAP database instead of a dataframe. I know SQL, so this is the easiest one to pick up.

There are far more options and this is my shortlist. I have excluded the other options in the benchmark for these reasons:

- Less popular than pandas (meaning less help available)
- Slower than polars (dask, Arrow, Modin)
- Not mature enough, as shown by lower activity on Github (siuba, pydatatable)
- Requires other software or a server (ClickHouse)
- Not in Python (DataFrames.jl)
- Meant for GPU only (cuDF)

My reference is my current use of [dplyr](https://dplyr.tidyverse.org) in R. When I need more performance, I use [tidytable](https://github.com/markfairbanks/tidytable) to get the speed of data.table with the grammar of dplyr. I also use [dbplyr](https://dbplyr.tidyverse.org) a lot, which translates dplyr to SQL. It's composeable, which actually makes it superior to SQL for me in most use cases.

With that out of the way, here's a heavily biased comparison of the three Python packages.

I'm speaking of my personal opinion of these packages given my own background - not a general comparison.

## Pandas: Most popular

The syntax is inspired by base R, which is a good thing.

Pandas uses a row index, which is basically a special blessed column. Base R also has this with row names, though the tidyverse and tibbles have largely removed them from common use.

Pandas has the widest API, offering hundreds of functions for every conceivable manipulation.

## Polars: Fastest

Polars is written in Rust and also offers a Python API. It comes in two flavors: eager and lazy. Lazy evaluation is similar to how dbplyr and dtplyer work: until asked, nothing is evaluated. This enables performance gains by reordering the commands being executed. But it's a little less convenient for interactive analysis.

The API is leaner than pandas, requiring to memorize fewer functions and patterns.

There isn't nearly as much help available for problems with polars as for with pandas. Often, I had to use trial and error based on the documentation. While the documentation is good, it can't answer every question.

## DuckDB: Easiest for SQL users

DuckDB can also operate directly on a pandas dataframe.

The performance is closer to polars than to pandas.

A big plus is the ability to handle larger than memory data.

The code is portable to R, C, C++, Java and other programming languages the duckdb has [APIs](https://duckdb.org/docs/api/overview). It's also portable when the logic is taken to a DB like [Postgres](https://www.postgresql.org), or [Snowflake](https://www.snowflake.com/), or is ported to an ETL framework like [DBT](https://github.com/dbt-labs/dbt-core).

This stands in contrast to polars and pandas code, which has to be rewritten from scratch. It also means that the skill gained in manipulating data translates well to other situations - SQL has been around for more than 40 years. Something that can't be said about any Python library. Learning SQL is future-proofing ones career.

While these are big plusses, duckdb isn't as convenient as Polars and Pandas for interactive data exploration. The SQL isn't as composable. Plus, writing strings rather than actual Python is awkward and many editors don't provide syntax highlighting within the strings (Jetbrains editors like [PyCharm](https://www.jetbrains.com/pycharm/) and [DataSpell](https://www.jetbrains.com/dataspell/) do).

SQL is less expressive than Python, especially when the names of output columns are unknown. It lacks shorthands.

It's also harder to write custom functions in SQL. With pandas and polars, custom operations are just one lambda away.

Using duckdb without pandas doesn't seem feasible for exploratory data analysis, because graphing packages like seaborn and plotly expect a pandas data frame or similar as an input.

Speed of iteration is critical: the faster one can iterate, the more hypotheses about the data can be tested.

## Conclusion

It's not a clear-cut choice. Each seems more useful in it's own arena.

None of the three options offer a syntax that is as convenient for interactive analysis as dplyr. Polars is the closest to it, but dplyr still has an edge with [tidy evaluation](https://www.tidyverse.org/blog/2019/06/rlang-0-4-0/#a-simpler-interpolation-pattern-with), letting users refer to columns in a data frame by their names (`colname`) rather than as strings `"colname"`or constructs like `pl.col("colname")`. While this is nice for quickly writing code, I've also seen it be confusing for newbies to R that mix it up with base R's syntax. It's also harder to program with, where it's necessary to use operators like `{{ }}` and `:=`.

Photo by <a href="https://unsplash.com/@hharritt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Hunter Harritt</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>