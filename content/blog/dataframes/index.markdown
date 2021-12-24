---
title: "Data frame wars: pandas vs. polars vs duckdb"
excerpt: "A comparison of pandas, polars and duckdb from the perspective of a dplyr user."
slug: "dataframes"
author: "Paul Simmering"
date: "2021-12-20"
categories: ["R", "Python"]
tags: ["Performance"]
---

<script src="{{< blogdown/postref >}}index_files/core-js/shim.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/react/react.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/react/react-dom.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/reactwidget/react-tools.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<script src="{{< blogdown/postref >}}index_files/reactable-binding/reactable.js"></script>
<script src="{{< blogdown/postref >}}index_files/core-js/shim.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/react/react.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/react/react-dom.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/reactwidget/react-tools.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<script src="{{< blogdown/postref >}}index_files/reactable-binding/reactable.js"></script>
<script src="{{< blogdown/postref >}}index_files/core-js/shim.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/react/react.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/react/react-dom.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/reactwidget/react-tools.js"></script>
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<script src="{{< blogdown/postref >}}index_files/reactable-binding/reactable.js"></script>

I’m a long time R user and lately I’ve seen more and more signals that it’s worth investing into learning more Python. I use it for NLP with [spaCy](https://spacy.io) and to build functions on [AWS Lambda](https://aws.amazon.com/lambda/features/) (though the recent [lambdr](https://mdneuzerling.com/post/serverless-on-demand-parametrised-r-markdown-reports-with-aws-lambda/) package gave R some great tools there too). Further, there are many more data API libraries available for Python than for R.

## The contenders

Python has a larger package ecosystem and ways of doing things than R, which seems more centralized thanks to CRAN. Adopting Python means making many choices on which libraries to invest time into learning. The first thing I needed was a way to manipulate data frames. I’ve narrowed it down to 3 choices:

-   [Pandas](https://pandas.pydata.org): The most commonly used library and the one with the most tutorials and Stack Overflow answers available.
-   [Polars](https://www.pola.rs): The fastest library available. It’s a new library that doesn’t have nearly as many users or help available. But according to the [H2Oai ops benchmark](https://h2oai.github.io/db-benchmark/), it often runs 10x faster than Pandas.
-   [Duckdb](https://www.pola.rs): Use an in-memory OLAP database instead of a dataframe. I know SQL, so this is the easiest one to pick up.

There are far more options and this is my shortlist. I have excluded the other options in the benchmark for these reasons:

-   Less popular than pandas (meaning less help available)
-   Slower than polars (dask, Arrow, Modin)
-   Not mature enough, as shown by lower activity on Github (siuba, pydatatable)
-   Requires other software or a server (ClickHouse)
-   Not in Python (DataFrames.jl)
-   Meant for GPU only (cuDF)

My reference is my current use of [dplyr](https://dplyr.tidyverse.org) in R. When I need more performance, I use [tidytable](https://github.com/markfairbanks/tidytable) to get the speed of data.table with the grammar of dplyr. I also use [dbplyr](https://dbplyr.tidyverse.org) a lot, which translates dplyr to SQL. It’s composeable, which actually makes it superior to SQL for me in most use cases.

With that out of the way, here’s a heavily biased comparison of the three Python packages.

I’m speaking of my personal opinion of these packages given my own background - not a general comparison. I’ll compare the three contenders by running a data transformation pipeline involving import from CSV, mutate, filter, sort, join, group by and summarise. I’ll use the nycflights13 dataset, which some readers may know from Hadley Wickham’s [R for Data Science](https://r4ds.had.co.nz/transform.html).

## dplyr: Reference in R

Let’s start off the comparison with a reference implementation with my current default, dplyr. The dataset is available as a package, so I skip the CSV import step here.

``` r
suppressMessages(library(dplyr))
library(nycflights13)
library(reactable)

# Take a look at the tables
reactable(head(flights, 10))
```

<div id="htmlwidget-1" class="reactable html-widget" style="width:auto;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"tag":{"name":"Reactable","attribs":{"data":{"year":[2013,2013,2013,2013,2013,2013,2013,2013,2013,2013],"month":[1,1,1,1,1,1,1,1,1,1],"day":[1,1,1,1,1,1,1,1,1,1],"dep_time":[517,533,542,544,554,554,555,557,557,558],"sched_dep_time":[515,529,540,545,600,558,600,600,600,600],"dep_delay":[2,4,2,-1,-6,-4,-5,-3,-3,-2],"arr_time":[830,850,923,1004,812,740,913,709,838,753],"sched_arr_time":[819,830,850,1022,837,728,854,723,846,745],"arr_delay":[11,20,33,-18,-25,12,19,-14,-8,8],"carrier":["UA","UA","AA","B6","DL","UA","B6","EV","B6","AA"],"flight":[1545,1714,1141,725,461,1696,507,5708,79,301],"tailnum":["N14228","N24211","N619AA","N804JB","N668DN","N39463","N516JB","N829AS","N593JB","N3ALAA"],"origin":["EWR","LGA","JFK","JFK","LGA","EWR","EWR","LGA","JFK","LGA"],"dest":["IAH","IAH","MIA","BQN","ATL","ORD","FLL","IAD","MCO","ORD"],"air_time":[227,227,160,183,116,150,158,53,140,138],"distance":[1400,1416,1089,1576,762,719,1065,229,944,733],"hour":[5,5,5,5,6,5,6,6,6,6],"minute":[15,29,40,45,0,58,0,0,0,0],"time_hour":["2013-01-01T05:00:00","2013-01-01T05:00:00","2013-01-01T05:00:00","2013-01-01T05:00:00","2013-01-01T06:00:00","2013-01-01T05:00:00","2013-01-01T06:00:00","2013-01-01T06:00:00","2013-01-01T06:00:00","2013-01-01T06:00:00"]},"columns":[{"accessor":"year","name":"year","type":"numeric"},{"accessor":"month","name":"month","type":"numeric"},{"accessor":"day","name":"day","type":"numeric"},{"accessor":"dep_time","name":"dep_time","type":"numeric"},{"accessor":"sched_dep_time","name":"sched_dep_time","type":"numeric"},{"accessor":"dep_delay","name":"dep_delay","type":"numeric"},{"accessor":"arr_time","name":"arr_time","type":"numeric"},{"accessor":"sched_arr_time","name":"sched_arr_time","type":"numeric"},{"accessor":"arr_delay","name":"arr_delay","type":"numeric"},{"accessor":"carrier","name":"carrier","type":"character"},{"accessor":"flight","name":"flight","type":"numeric"},{"accessor":"tailnum","name":"tailnum","type":"character"},{"accessor":"origin","name":"origin","type":"character"},{"accessor":"dest","name":"dest","type":"character"},{"accessor":"air_time","name":"air_time","type":"numeric"},{"accessor":"distance","name":"distance","type":"numeric"},{"accessor":"hour","name":"hour","type":"numeric"},{"accessor":"minute","name":"minute","type":"numeric"},{"accessor":"time_hour","name":"time_hour","type":"Date"}],"defaultPageSize":10,"paginationType":"numbers","showPageInfo":true,"minRows":1,"dataKey":"0fe230bb41adb4c97e1205322b49f222","key":"0fe230bb41adb4c97e1205322b49f222"},"children":[]},"class":"reactR_markup"},"evals":[],"jsHooks":[]}</script>
reactable(head(airlines, 10))
<div id="htmlwidget-2" class="reactable html-widget" style="width:auto;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-2">{"x":{"tag":{"name":"Reactable","attribs":{"data":{"carrier":["9E","AA","AS","B6","DL","EV","F9","FL","HA","MQ"],"name":["Endeavor Air Inc.","American Airlines Inc.","Alaska Airlines Inc.","JetBlue Airways","Delta Air Lines Inc.","ExpressJet Airlines Inc.","Frontier Airlines Inc.","AirTran Airways Corporation","Hawaiian Airlines Inc.","Envoy Air"]},"columns":[{"accessor":"carrier","name":"carrier","type":"character"},{"accessor":"name","name":"name","type":"character"}],"defaultPageSize":10,"paginationType":"numbers","showPageInfo":true,"minRows":1,"dataKey":"08ac76438f639beb63f40296e9c2c3c3","key":"08ac76438f639beb63f40296e9c2c3c3"},"children":[]},"class":"reactR_markup"},"evals":[],"jsHooks":[]}</script>

The `flights` tables has 336776 rows, one for each flight of an airplane. The `airlines` table has 16 rows, one for each airline mapping the full name of the company to a shortcode.

Let’s find the airline with the highest arrival delays in January 2013. Some values in `arr_delay` are negative, indicating that the flight was faster than expected. I replace these values with 0 because I don’t want them to cancel out delays of other flights. I join to the airlines table to get the full names of the airlines.

``` r
delayed_airlines <- flights |>
  filter(year == 2013, month == 1) |> 
  mutate(arr_delay = replace(arr_delay, arr_delay < 0, 0)) |>
  left_join(airlines, by = "carrier") |>
  group_by(airline = name) |>
  summarise(
    flights = n(),
    mean_delay = mean(arr_delay, na.rm = TRUE)
  ) %>% 
  arrange(desc(mean_delay))

reactable(delayed_airlines)
```

<div id="htmlwidget-3" class="reactable html-widget" style="width:auto;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-3">{"x":{"tag":{"name":"Reactable","attribs":{"data":{"airline":["SkyWest Airlines Inc.","Hawaiian Airlines Inc.","ExpressJet Airlines Inc.","Frontier Airlines Inc.","Mesa Airlines Inc.","Endeavor Air Inc.","Alaska Airlines Inc.","Envoy Air","Southwest Airlines Co.","JetBlue Airways","United Air Lines Inc.","American Airlines Inc.","AirTran Airways Corporation","US Airways Inc.","Delta Air Lines Inc.","Virgin America"],"flights":[1,31,4171,59,46,1573,62,2271,996,4427,4637,2794,328,1602,3690,316],"mean_delay":[107,48.7741935483871,29.6427850655903,23.8813559322034,20.4102564102564,19.3216216216216,17.6451612903226,14.3036768043577,12.9644670050761,12.9193292544754,11.8518518518519,10.9533773861968,9.9537037037037,9.11132561132561,8.0703146374829,3.1656050955414]},"columns":[{"accessor":"airline","name":"airline","type":"character"},{"accessor":"flights","name":"flights","type":"numeric"},{"accessor":"mean_delay","name":"mean_delay","type":"numeric"}],"defaultPageSize":10,"paginationType":"numbers","showPageInfo":true,"minRows":1,"dataKey":"e869ab85166b7808dd0d42af69c19dd8","key":"e869ab85166b7808dd0d42af69c19dd8"},"children":[]},"class":"reactR_markup"},"evals":[],"jsHooks":[]}</script>

I export two tables from the dataset to CSV to make it available for Python packages.

``` r
# Use data.table::fwrite instead of write.csv because it's faster
data.table::fwrite(flights, "flights.csv", row.names = FALSE)
data.table::fwrite(airlines, "airlines.csv", row.names = FALSE)
```

## Pandas: Most popular

The syntax is inspired by base R, which is a good thing.

``` python
import pandas as pd

# Import from CSV
flights = pd.read_csv("flights.csv")
airlines = pd.read_csv("airlines.csv")
```

`pandas.read_csv` read the header and conveniently inferred the column types.

``` python
(
  flights
    .assign(arr_delay = flights.arr_delay.clip(lower = 0))
    .query("year == 2013 & month == 1")
    .merge(airlines, how = "left", on = "carrier")
    .rename(columns = {"name": "airline"})
    .groupby("airline")
    .agg({'airline':'count', 'arr_delay':'mean'})
    .rename(columns = {"airline": "flights", "arr_delay": "mean_delay"})
    .sort_values(by= "mean_delay", ascending = False)
)
##                              flights  mean_delay
## airline                                         
## SkyWest Airlines Inc.              1  107.000000
## Hawaiian Airlines Inc.            31   48.774194
## ExpressJet Airlines Inc.        4171   29.642785
## Frontier Airlines Inc.            59   23.881356
## Mesa Airlines Inc.                46   20.410256
## Endeavor Air Inc.               1573   19.321622
## Alaska Airlines Inc.              62   17.645161
## Envoy Air                       2271   14.303677
## Southwest Airlines Co.           996   12.964467
## JetBlue Airways                 4427   12.919329
## United Air Lines Inc.           4637   11.851852
## American Airlines Inc.          2794   10.953377
## AirTran Airways Corporation      328    9.953704
## US Airways Inc.                 1602    9.111326
## Delta Air Lines Inc.            3690    8.070315
## Virgin America                   316    3.165605
```

Rows with missing values for `arr_delay` are dropped implicitly in the `agg` step.

Pandas uses a row index, which is basically a special column. Base R also has this with row names, though the tidyverse and tibbles have largely removed them from common use.

Pandas has the widest API, offering hundreds of functions for every conceivable manipulation.

## Polars: Fastest

Polars is written in Rust and also offers a Python API. It comes in two flavors: eager and lazy. Lazy evaluation is similar to how dbplyr and dtplyer work: until asked, nothing is evaluated. This enables performance gains by reordering the commands being executed. But it’s a little less convenient for interactive analysis. I’ll use the eager API here.

``` python
import polars as pl

# Import from CSV
```

The API is leaner than pandas, requiring to memorize fewer functions and patterns.

There isn’t nearly as much help available for problems with polars as for with pandas. Often, I had to use trial and error based on the documentation. While the documentation is good, it can’t answer every question.

## DuckDB: Easiest for SQL users

DuckDB can also operate directly on a pandas dataframe.

``` python
import duckdb

# Import from CSV
"""
CREATE TABLE 'flights' AS
SELECT * FROM 'flights.csv'
"""
## "\nCREATE TABLE 'flights' AS\nSELECT * FROM 'flights.csv'\n"
```

The performance is closer to polars than to pandas.

A big plus is the ability to handle larger than memory data.

The code is portable to R, C, C++, Java and other programming languages the duckdb has [APIs](https://duckdb.org/docs/api/overview). It’s also portable when the logic is taken to a DB like [Postgres](https://www.postgresql.org), or [Snowflake](https://www.snowflake.com/), or is ported to an ETL framework like [DBT](https://github.com/dbt-labs/dbt-core).

This stands in contrast to polars and pandas code, which has to be rewritten from scratch. It also means that the skill gained in manipulating data translates well to other situations - SQL has been around for more than 40 years. Something that can’t be said about any Python library. Learning SQL is future-proofing ones career.

While these are big plusses, duckdb isn’t as convenient as Polars and Pandas for interactive data exploration. The SQL isn’t as composable. Plus, writing strings rather than actual Python is awkward and many editors don’t provide syntax highlighting within the strings (Jetbrains editors like [PyCharm](https://www.jetbrains.com/pycharm/) and [DataSpell](https://www.jetbrains.com/dataspell/) do).

SQL is less expressive than Python, especially when the names of output columns are unknown. It lacks shorthands.

It’s also harder to write custom functions in SQL. With pandas and polars, custom operations are just one lambda away.

Using duckdb without pandas doesn’t seem feasible for exploratory data analysis, because graphing packages like seaborn and plotly expect a pandas data frame or similar as an input.

Speed of iteration is critical: the faster one can iterate, the more hypotheses about the data can be tested.

## Conclusion

It’s not a clear-cut choice. Each seems more useful in it’s own arena.

None of the three options offer a syntax that is as convenient for interactive analysis as dplyr. Polars is the closest to it, but dplyr still has an edge with [tidy evaluation](https://www.tidyverse.org/blog/2019/06/rlang-0-4-0/#a-simpler-interpolation-pattern-with), letting users refer to columns in a data frame by their names (`colname`) rather than as strings `"colname"`or constructs like `pl.col("colname")`. While this is nice for quickly writing code, I’ve also seen it be confusing for newbies to R that mix it up with base R’s syntax. It’s also harder to program with, where it’s necessary to use operators like `{{ }}` and `:=`.

Personally, I’ll leverage my existing knowledge and rely on SQL and an OLAP database (such as Snowflake) to do the heavy lifting. For steps that are better done locally, I’ll use pandas for maximum compatibility. The syntax isn’t my favorite, but there’s so much online help available that StackOverflow has the answer for almost any problem. Github Copilot also deserves a mention for making it easier to pick up.

Photo by <a href="https://unsplash.com/@hharritt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Hunter Harritt</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
