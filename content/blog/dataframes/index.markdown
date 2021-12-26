---
title: "Data frame wars: Choosing a Python dataframe library as a dplyr user"
excerpt: "A comparison of pandas, siuba, pydatatable, polars and duckdb from the perspective of a dplyr user"
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

I’m a long time R user and lately I’ve seen more and more [signals](https://www.tiobe.com/tiobe-index/python/) that it’s worth investing into Python. I use it for NLP with [spaCy](https://spacy.io) and to build functions on [AWS Lambda](https://aws.amazon.com/lambda/features/). Further, there are many more data API libraries and machine learning libraries for Python than for R.

Adopting Python means making choices on which libraries to invest time into learning. Manipulating data frames is one of the most common data science activities, so choosing the right library for it is key.

Michael Chow, developer of [siuba](https://github.com/machow/siuba), a Python port of dplyr on top of pandas [wrote](https://mchow.com/posts/pandas-has-a-hard-job/) describes the situation well:

> It seems like there’s been a lot of frustration surfacing on twitter lately from people coming from R—especially if they’ve used dplyr and ggplot—towards pandas and matplotlib. I can relate. I’m developing a port of dplyr to python. But in the end, it’s probably helpful to view these libraries as foundational to a lot of other, higher-level libraries (some of which will hopefully get things right for you!).

The higher-level libraries he mentions come with a problem : There’s no universal standard.

In a discussion of the polars library on Hacker News the user “civilized” put the dplyr user perspective more bluntly:

> In my world, anything that isn’t “identical to R’s dplyr API but faster” just isn’t quite worth switching for. There’s absolutely no contest: dplyr has the most productive API and that matters to me more than anything else.

I’m more willing to compromise though, so here’s a comparison of the strongest contenders.

## The contenders

The [database-like ops benchmark on H2Oai](https://h2oai.github.io/db-benchmark/) is a helpful performance comparison.

I’m considering these libraries:

1.  [Pandas](https://pandas.pydata.org): The most commonly used library and the one with the most tutorials and Stack Overflow answers available.
2.  [siuba](https://github.com/machow/siuba): A port of dplyr to Python, built on top of pandas. Not in the benchmark. Performance probably similar to pandas or worse due to translation.
3.  [Polars](https://www.pola.rs): The fastest library available. According to the benchmark, it runs 3-10x faster than Pandas.
4.  [Duckdb](https://www.pola.rs): Use an in-memory OLAP database instead of a dataframe and write SQL. In R, this can also be queried via dbplyr.
5.  [ibis](https://ibis-project.org/docs/index.html). Backend-agnostic wrapper for pandas and SQL engines.

There are more options. I excluded the others for these reasons:

-   Slower than polars and not with a readability focus (dask, Arrow, Modin, pydatatable)
-   Requires or is optmized for running on a remote server (Spark, ClickHouse and most other SQL databases).
-   Not meant for OLAP (sqlite)
-   Not in Python (DataFrames.jl)
-   Meant for GPU (cuDF)

The benchmark provides a comparison of performance, but another important factor is popularity and maturity. A more mature library has a more stable API, better test coverage and there is more help available online, such as on StackOverflow. One way to measure popularity is the number of stars that the package repository has on Github.

``` r
library(ggplot2)
libs <- data.frame(
  library = c("pandas", "siuba", "polars", "duckdb", "dplyr", "data.table", "pydatatable", "dtplyr", "tidytable", "ibis"),
  language = c("Python", "Python", "Python", "SQL", "R", "R", "Python", "R", "R", "Python"),
  stars = c(32100, 732, 3900, 4100, 3900, 2900, 1400, 542, 285, 1600)
)

ggplot(libs, aes(x = reorder(library, -stars), y = stars, fill = language)) +
  geom_col() + 
  labs(
    title = "Pandas is by far the most popular choice",
    subtitle = "Comparison of Github stars on 2021-12-25",
    fill = "Language",
    x = "Library",
    y = "Github stars"
  )
```

<img src="{{< blogdown/postref >}}index_files/figure-html/github_stars-1.png" width="672" />

Github stars are not a perfect proxy. For instance, dplyr is more mature than its star count suggests. Comparing the completeness of the documentation and tutorials for dplyr and polars reveals that it’s a day and night difference.

With the quantitative comparison out of the way, here’s a qualitative comparison of the Python packages. I’m speaking of my personal opinion of these packages - not a general comparison. My reference is my current use of [dplyr](https://dplyr.tidyverse.org) in R. When I need more performance, I use [tidytable](https://github.com/markfairbanks/tidytable) to get most of the speed of data.table with the grammar of dplyr and eager evaluation. Another alternative is [dtplyr](https://github.com/tidyverse/dtplyr), which translates dplyr to data.table with lazy evaluation. I also use [dbplyr](https://dbplyr.tidyverse.org), which translates dplyr to SQL.

I’ll compare the libraries by running a data transformation pipeline involving import from CSV, mutate, filter, sort, join, group by and summarize. I’ll use the nycflights13 dataset, which is featured in Hadley Wickham’s [R for Data Science](https://r4ds.had.co.nz/transform.html).

## dplyr: Reference in R

Let’s start with a reference implementation in dplyr. The dataset is available as a package, so I skip the CSV import.

``` r
library(dplyr, warn.conflicts = FALSE)
library(nycflights13)
library(reactable)

# Take a look at the tables
reactable(head(flights, 10))
```

<div id="htmlwidget-1" class="reactable html-widget" style="width:auto;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"tag":{"name":"Reactable","attribs":{"data":{"year":[2013,2013,2013,2013,2013,2013,2013,2013,2013,2013],"month":[1,1,1,1,1,1,1,1,1,1],"day":[1,1,1,1,1,1,1,1,1,1],"dep_time":[517,533,542,544,554,554,555,557,557,558],"sched_dep_time":[515,529,540,545,600,558,600,600,600,600],"dep_delay":[2,4,2,-1,-6,-4,-5,-3,-3,-2],"arr_time":[830,850,923,1004,812,740,913,709,838,753],"sched_arr_time":[819,830,850,1022,837,728,854,723,846,745],"arr_delay":[11,20,33,-18,-25,12,19,-14,-8,8],"carrier":["UA","UA","AA","B6","DL","UA","B6","EV","B6","AA"],"flight":[1545,1714,1141,725,461,1696,507,5708,79,301],"tailnum":["N14228","N24211","N619AA","N804JB","N668DN","N39463","N516JB","N829AS","N593JB","N3ALAA"],"origin":["EWR","LGA","JFK","JFK","LGA","EWR","EWR","LGA","JFK","LGA"],"dest":["IAH","IAH","MIA","BQN","ATL","ORD","FLL","IAD","MCO","ORD"],"air_time":[227,227,160,183,116,150,158,53,140,138],"distance":[1400,1416,1089,1576,762,719,1065,229,944,733],"hour":[5,5,5,5,6,5,6,6,6,6],"minute":[15,29,40,45,0,58,0,0,0,0],"time_hour":["2013-01-01T05:00:00","2013-01-01T05:00:00","2013-01-01T05:00:00","2013-01-01T05:00:00","2013-01-01T06:00:00","2013-01-01T05:00:00","2013-01-01T06:00:00","2013-01-01T06:00:00","2013-01-01T06:00:00","2013-01-01T06:00:00"]},"columns":[{"accessor":"year","name":"year","type":"numeric"},{"accessor":"month","name":"month","type":"numeric"},{"accessor":"day","name":"day","type":"numeric"},{"accessor":"dep_time","name":"dep_time","type":"numeric"},{"accessor":"sched_dep_time","name":"sched_dep_time","type":"numeric"},{"accessor":"dep_delay","name":"dep_delay","type":"numeric"},{"accessor":"arr_time","name":"arr_time","type":"numeric"},{"accessor":"sched_arr_time","name":"sched_arr_time","type":"numeric"},{"accessor":"arr_delay","name":"arr_delay","type":"numeric"},{"accessor":"carrier","name":"carrier","type":"character"},{"accessor":"flight","name":"flight","type":"numeric"},{"accessor":"tailnum","name":"tailnum","type":"character"},{"accessor":"origin","name":"origin","type":"character"},{"accessor":"dest","name":"dest","type":"character"},{"accessor":"air_time","name":"air_time","type":"numeric"},{"accessor":"distance","name":"distance","type":"numeric"},{"accessor":"hour","name":"hour","type":"numeric"},{"accessor":"minute","name":"minute","type":"numeric"},{"accessor":"time_hour","name":"time_hour","type":"Date"}],"defaultPageSize":10,"paginationType":"numbers","showPageInfo":true,"minRows":1,"dataKey":"0fe230bb41adb4c97e1205322b49f222","key":"0fe230bb41adb4c97e1205322b49f222"},"children":[]},"class":"reactR_markup"},"evals":[],"jsHooks":[]}</script>

``` r
reactable(head(airlines, 10))
```

<div id="htmlwidget-2" class="reactable html-widget" style="width:auto;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-2">{"x":{"tag":{"name":"Reactable","attribs":{"data":{"carrier":["9E","AA","AS","B6","DL","EV","F9","FL","HA","MQ"],"name":["Endeavor Air Inc.","American Airlines Inc.","Alaska Airlines Inc.","JetBlue Airways","Delta Air Lines Inc.","ExpressJet Airlines Inc.","Frontier Airlines Inc.","AirTran Airways Corporation","Hawaiian Airlines Inc.","Envoy Air"]},"columns":[{"accessor":"carrier","name":"carrier","type":"character"},{"accessor":"name","name":"name","type":"character"}],"defaultPageSize":10,"paginationType":"numbers","showPageInfo":true,"minRows":1,"dataKey":"08ac76438f639beb63f40296e9c2c3c3","key":"08ac76438f639beb63f40296e9c2c3c3"},"children":[]},"class":"reactR_markup"},"evals":[],"jsHooks":[]}</script>

The `flights` tables has 336776 rows, one for each flight of an airplane. The `airlines` table has 16 rows, one for each airline mapping the full name of the company to a code.

Let’s find the airline with the highest arrival delays in January 2013.

``` r
flights |>
  filter(year == 2013, month == 1, !is.na(arr_delay)) |> 
  mutate(arr_delay = replace(arr_delay, arr_delay < 0, 0)) |>
  left_join(airlines, by = "carrier") |>
  group_by(airline = name) |>
  summarise(flights = n(), mean_delay = mean(arr_delay)) |> 
  arrange(desc(mean_delay))
```

    ## # A tibble: 16 × 3
    ##    airline                     flights mean_delay
    ##    <chr>                         <int>      <dbl>
    ##  1 SkyWest Airlines Inc.             1     107   
    ##  2 Hawaiian Airlines Inc.           31      48.8 
    ##  3 ExpressJet Airlines Inc.       3964      29.6 
    ##  4 Frontier Airlines Inc.           59      23.9 
    ##  5 Mesa Airlines Inc.               39      20.4 
    ##  6 Endeavor Air Inc.              1480      19.3 
    ##  7 Alaska Airlines Inc.             62      17.6 
    ##  8 Envoy Air                      2203      14.3 
    ##  9 Southwest Airlines Co.          985      13.0 
    ## 10 JetBlue Airways                4413      12.9 
    ## 11 United Air Lines Inc.          4590      11.9 
    ## 12 American Airlines Inc.         2724      11.0 
    ## 13 AirTran Airways Corporation     324       9.95
    ## 14 US Airways Inc.                1554       9.11
    ## 15 Delta Air Lines Inc.           3655       8.07
    ## 16 Virgin America                  314       3.17

Some values in `arr_delay` are negative, indicating that the flight was faster than expected. I replaced these values with 0 because I don’t want them to cancel out delays of other flights. I joined to the airlines table to get the full names of the airlines.

I export the flights and airlines tables to CSV to hand them over to Python.

``` r
data.table::fwrite(flights, "flights.csv", row.names = FALSE)
data.table::fwrite(airlines, "airlines.csv", row.names = FALSE)
```

## Pandas: Most popular

The following sections follow a pattern: read in from CSV, then build a query.

``` python
import pandas as pd

# Import from CSV
flights_pd = pd.read_csv("flights.csv")
airlines_pd = pd.read_csv("airlines.csv")
```

`pandas.read_csv` reads the header and conveniently infers the column types.

``` python
(flights_pd
  .query("year == 2013 & month == 1 & arr_delay.notnull()")
  .assign(arr_delay = flights_pd.arr_delay.clip(lower = 0))
  .merge(airlines_pd, how = "left", on = "carrier")
  .rename(columns = {"name": "airline"})
  .groupby("airline")
  .agg(flights = ("airline", "count"), mean_delay = ("arr_delay", "mean"))
  .sort_values(by = "mean_delay", ascending = False))
```

    ##                              flights  mean_delay
    ## airline                                         
    ## SkyWest Airlines Inc.              1  107.000000
    ## Hawaiian Airlines Inc.            31   48.774194
    ## ExpressJet Airlines Inc.        3964   29.642785
    ## Frontier Airlines Inc.            59   23.881356
    ## Mesa Airlines Inc.                39   20.410256
    ## Endeavor Air Inc.               1480   19.321622
    ## Alaska Airlines Inc.              62   17.645161
    ## Envoy Air                       2203   14.303677
    ## Southwest Airlines Co.           985   12.964467
    ## JetBlue Airways                 4413   12.919329
    ## United Air Lines Inc.           4590   11.851852
    ## American Airlines Inc.          2724   10.953377
    ## AirTran Airways Corporation      324    9.953704
    ## US Airways Inc.                 1554    9.111326
    ## Delta Air Lines Inc.            3655    8.070315
    ## Virgin America                   314    3.165605

I chose to use the pipeline syntax from pandas - another option is to modify the dataset in place. That has a lower memory footprint, but can’t be run repeatedly for the same result, such as in interactive use in a notebook.

Here, the `query()` function is slightly awkward with the long string argument. The `groupby` doesn’t allow renaming on the fly like dplyr, though I don’t consider that a real drawback. Perhaps it’s clearer to rename explicitly anyway.

Pandas has the widest API, offering hundreds of functions for every conceivable manipulation. The `clip` function used here is one such example. One difference to dplyr is that pandas uses its own methods `.mean()`, rather than using external ones such as `base::mean()`. That means using custom functions instead carries a [performance penalty](https://stackoverflow.com/a/26812998).

As we’ll see later, pandas is the backend for siuba and ibis, which boil down to pandas code.

One difference to all other discussed solutions is that pandas uses a [row index](https://www.sharpsightlabs.com/blog/pandas-index/). Base R also has this with row names, but the tidyverse and tibbles have largely removed them from common use. I never missed row names. At the times I had to work with them in pandas they were more confusing than helpful. The documentation of polars puts it more bluntly:

> No index. They are not needed. Not having them makes things easier. Convince me otherwise

That’s quite passive aggressive, but I do agree and wish pandas didn’t have it.

## siuba: dplyr in Python

``` python
import siuba as si

# Import from CSV
flights_si = pd.read_csv("flights.csv")
airlines_si = pd.read_csv("airlines.csv")
```

As siuba is just an alternative way of writing some pandas commands, we read the data just like in the pandas implementation.

``` python
(
  flights_si
    >> si.filter(
      si._.year == 2013,
      si._.month == 1,
      si._.arr_delay.notnull()
    )
    >> si.mutate(arr_delay = si._.arr_delay.clip(lower = 0))
    >> si.left_join(si._, airlines_si, on = "carrier")
    >> si.rename(airline = si._.name)
    >> si.group_by(si._.airline)
    >> si.summarize(
        flights = si._.airline.count(),
        mean_delay = si._.arr_delay.mean()
    )
    >> si.arrange(-si._.mean_delay)
)
```

    ##                         airline  flights  mean_delay
    ## 11        SkyWest Airlines Inc.        1  107.000000
    ## 8        Hawaiian Airlines Inc.       31   48.774194
    ## 6      ExpressJet Airlines Inc.     3964   29.642785
    ## 7        Frontier Airlines Inc.       59   23.881356
    ## 10           Mesa Airlines Inc.       39   20.410256
    ## 4             Endeavor Air Inc.     1480   19.321622
    ## 1          Alaska Airlines Inc.       62   17.645161
    ## 5                     Envoy Air     2203   14.303677
    ## 12       Southwest Airlines Co.      985   12.964467
    ## 9               JetBlue Airways     4413   12.919329
    ## 14        United Air Lines Inc.     4590   11.851852
    ## 2        American Airlines Inc.     2724   10.953377
    ## 0   AirTran Airways Corporation      324    9.953704
    ## 13              US Airways Inc.     1554    9.111326
    ## 3          Delta Air Lines Inc.     3655    8.070315
    ## 15               Virgin America      314    3.165605

I found siuba the easiest to work with. Once I understood the `_` placeholder for a table of data, I could write it almost as fast as dplyr. Out of all the ways to refer to a column in a data frame, I found it to be the most convenient, because it doesn’t require me to spell out the name of the data frame over and over. While not as elegant as dplyr’s [tidy evaluation](https://www.tidyverse.org/blog/2019/06/rlang-0-4-0/#a-simpler-interpolation-pattern-with) (discussed at the end of the article), it avoids the ambivalence in dplyr where it can be unclear whether a name refers to a column or an outside object.

It’s always possible to drop into pandas, such as for the aggregation functions which use the `mean()` and `count()` methods of the pandas series. The `>>` is an easy replacement for the `%>%` magrittr pipe or `|>` base pipe in R.

The author advertises siuba like this (from the [docs](https://siuba.readthedocs.io/en/latest/)):

> Siuba is a library for quick, scrappy data analysis in Python. It is a port of dplyr, tidyr, and other R Tidyverse libraries.

A way for dplyr users to quickly hack away at data analysis in Python, but not meant for unsupervised production use.

## Polars: Fastest

Polars is written in Rust and also offers a Python API. It comes in two flavors: eager and lazy. Lazy evaluation is similar to how dbplyr and dtplyr work: until asked, nothing is evaluated. This enables performance gains by reordering the commands being executed. But it’s a little less convenient for interactive analysis. I’ll use the eager API here.

``` python
import polars as pl

# Import from CSV
flights_pl = pl.read_csv("flights.csv")
airlines_pl = pl.read_csv("airlines.csv")
```

``` python
(flights_pl
  .filter((pl.col("year") == 2013) & (pl.col("month") == 1))
  .drop_nulls("arr_delay")
  .join(airlines_pl, on = "carrier", how = "left")
  .with_columns(
    [
      pl.when(pl.col("arr_delay") > 0)
        .then(pl.col("arr_delay"))
        .otherwise(0)
        .alias("arr_delay"),
      pl.col("name").alias("airline")
    ]
  )
  .groupby("airline")
  .agg(
    [
      pl.count("airline").alias("flights"),
      pl.mean("arr_delay").alias("mean_delay")
    ]
  )
  .sort("mean_delay", reverse = True)
)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1 "class="dataframe ">
<thead>
<tr>
<th>
airline
</th>
<th>
flights
</th>
<th>
mean_delay
</th>
</tr>
<tr>
<td>
str
</td>
<td>
u32
</td>
<td>
f64
</td>
</tr>
</thead>
<tbody>
<tr>
<td>
"SkyWest Airlines Inc."
</td>
<td>
1
</td>
<td>
107
</td>
</tr>
<tr>
<td>
"Hawaiian Airlines Inc."
</td>
<td>
31
</td>
<td>
48.774193548387096
</td>
</tr>
<tr>
<td>
"ExpressJet Airlines Inc."
</td>
<td>
3964
</td>
<td>
29.642785065590314
</td>
</tr>
<tr>
<td>
"Frontier Airlines Inc."
</td>
<td>
59
</td>
<td>
23.88135593220339
</td>
</tr>
<tr>
<td>
"Mesa Airlines Inc."
</td>
<td>
39
</td>
<td>
20.41025641025641
</td>
</tr>
<tr>
<td>
"Endeavor Air Inc."
</td>
<td>
1480
</td>
<td>
19.32162162162162
</td>
</tr>
<tr>
<td>
"Alaska Airlines Inc."
</td>
<td>
62
</td>
<td>
17.64516129032258
</td>
</tr>
<tr>
<td>
"Envoy Air"
</td>
<td>
2203
</td>
<td>
14.303676804357695
</td>
</tr>
<tr>
<td>
"Southwest Airlines Co."
</td>
<td>
985
</td>
<td>
12.964467005076141
</td>
</tr>
<tr>
<td>
"JetBlue Airways"
</td>
<td>
4413
</td>
<td>
12.919329254475414
</td>
</tr>
<tr>
<td>
"United Air Lines Inc."
</td>
<td>
4590
</td>
<td>
11.851851851851851
</td>
</tr>
<tr>
<td>
"American Airlines Inc."
</td>
<td>
2724
</td>
<td>
10.95337738619677
</td>
</tr>
<tr>
<td>
"AirTran Airways Corporation"
</td>
<td>
324
</td>
<td>
9.953703703703704
</td>
</tr>
<tr>
<td>
"US Airways Inc."
</td>
<td>
1554
</td>
<td>
9.111325611325611
</td>
</tr>
<tr>
<td>
"Delta Air Lines Inc."
</td>
<td>
3655
</td>
<td>
8.0703146374829
</td>
</tr>
<tr>
<td>
"Virgin America"
</td>
<td>
314
</td>
<td>
3.1656050955414012
</td>
</tr>
</tbody>
</table>
</div>

The API is leaner than pandas, requiring to memorize fewer functions and patterns. Though this can also be seen as less feature-complete. Pandas, for example has a dedicated `clip` function.

There isn’t nearly as much help available for problems with polars as for with pandas. While the documentation is good, it can’t answer every question and lots of trial and error is needed.

A comparison of polars and pandas is available in the [polars documentation](https://pola-rs.github.io/polars-book/user-guide/coming_from_pandas.html?highlight=assign#column-assignment).

## DuckDB: Highly compatible and easy for SQL users

``` python
import duckdb

con_duckdb = duckdb.connect(database = ':memory:')

# Import from CSV
con_duckdb.execute(
  "CREATE TABLE 'flights' AS "
  "SELECT * FROM read_csv_auto('flights.csv', header = True);"
  "CREATE TABLE 'airlines' AS "
  "SELECT * FROM read_csv_auto('airlines.csv', header = True);"
)
```

    ## <duckdb.DuckDBPyConnection object at 0x120b20070>

DuckDB’s `read_csv_auto()` works just like the csv readers in Python.

``` python
con_duckdb.execute(
  "WITH flights_clipped AS ( "
  "SELECT carrier, CASE WHEN arr_delay > 0 THEN arr_delay ELSE 0 END AS arr_delay "
  "FROM flights "
  "WHERE year = 2013 AND month = 1 AND arr_delay IS NOT NULL"
  ")"
  "SELECT name AS airline, COUNT(*) AS flights, AVG(arr_delay) AS mean_delay "
  "FROM flights_clipped "
  "LEFT JOIN airlines ON flights_clipped.carrier = airlines.carrier "
  "GROUP BY name "
  "ORDER BY mean_delay DESC "
).fetchdf()
```

    ##                         airline  flights  mean_delay
    ## 0         SkyWest Airlines Inc.        1  107.000000
    ## 1        Hawaiian Airlines Inc.       31   48.774194
    ## 2      ExpressJet Airlines Inc.     3964   29.642785
    ## 3        Frontier Airlines Inc.       59   23.881356
    ## 4            Mesa Airlines Inc.       39   20.410256
    ## 5             Endeavor Air Inc.     1480   19.321622
    ## 6          Alaska Airlines Inc.       62   17.645161
    ## 7                     Envoy Air     2203   14.303677
    ## 8        Southwest Airlines Co.      985   12.964467
    ## 9               JetBlue Airways     4413   12.919329
    ## 10        United Air Lines Inc.     4590   11.851852
    ## 11       American Airlines Inc.     2724   10.953377
    ## 12  AirTran Airways Corporation      324    9.953704
    ## 13              US Airways Inc.     1554    9.111326
    ## 14         Delta Air Lines Inc.     3655    8.070315
    ## 15               Virgin America      314    3.165605

The performance is closer to polars than to pandas. A big plus is the ability to handle larger than memory data.

DuckDB can also operate directly on a pandas dataframe. The SQL code is portable to R, C, C++, Java and other programming languages the duckdb has [APIs](https://duckdb.org/docs/api/overview). It’s also portable when the logic is taken to a DB like [Postgres](https://www.postgresql.org), or [Clickhouse](https://clickhouse.com), or is ported to an ETL framework like [DBT](https://github.com/dbt-labs/dbt-core).

This stands in contrast to polars and pandas code, which has to be rewritten from scratch. It also means that the skill gained in manipulating SQL translates well to other situations. SQL has been around for more than 50 years - learning SQL is future-proofing a career.

While these are big plusses, duckdb isn’t so convenient for interactive data exploration. SQL isn’t as composeable. Composing SQL queries requires many common table expressions (CTEs, `WITH x AS (SELECT ...)`). Reusing them for other queries is not as easy as with Python. SQL is typically less expressive than Python. It lacks shorthands and it’s awkward when there are many columns. It’s also harder to write custom functions in SQL than in R or Python. This is the motivation for using libraries like pandas and dplyr. But SQL can actually do a surprising amount of things, as database expert Haki Benita explained in a [detailed article](https://hakibenita.com/sql-for-data-analysis).

Or in short, from the [documentation](https://ibis-project.org) of ibis:

> SQL is widely used and very convenient when writing simple queries. But as the complexity of operations grow, SQL can become very difficult to deal with.

Then, there’s the issue of how to actually write the SQL code. Writing strings rather than actual Python is awkward and many editors don’t provide syntax highlighting within the strings (Jetbrains editors like [PyCharm](https://www.jetbrains.com/pycharm/) and [DataSpell](https://www.jetbrains.com/dataspell/) do). The other option is writing `.sql` that have placeholders for parameters. That’s cleaner and allows using a linter, but is inconvenient for interactive use.

SQL is inherently lazily executed, because the query planner needs to take the whole query into account before starting computation. This enables performance gains. For interactive use, lazy evaluation is less convenient, because one can’t see the intermediate results at each step. Speed of iteration is critical: the faster one can iterate, the more hypotheses about the data can be tested.

There is a [programmatic way to construct queries](https://github.com/duckdb/duckdb/blob/master/examples/python/duckdb-python.py) for duckdb, designed to provide a [dbplyr alternative](https://github.com/duckdb/duckdb/issues/302) in Python. Unfortunately its documentation is sparse.

Using duckdb without pandas doesn’t seem feasible for exploratory data analysis, because graphing packages like seaborn and plotly expect a pandas data frame or similar as an input.

## ibis: Lingua franca in Python

The goal of ibis is to provide a universal language for working with data frames in Python, regardless of the backend that is used. It’s tagline is: *Write your analytics code once, run in everywhere*. This is similar to how dplyr can use SQL as a backend with dbplyr and data.table with dtplyr.

Among others, Ibis supports pandas, PostgreSQL and SQLite as backends. Unfortunately duckdb is not an available backend, because the authors of duckdb have [decided against](https://github.com/duckdb/duckdb/issues/302) building on ibis.

The ibis project aims to bridge the gap between the needs of interactive data analysis and the capabilities of SQL, which I have detailed in the previous section on duckdb.

For the test drive, I’ll use the [pandas backend](https://ibis-project.org/docs/backends/pandas.html), meaning that the ibis code is translated to pandas operations, similar to how siuba is translated to pandas.

``` python
import ibis
ibis.options.interactive = True

# Import from CSV into pandas
flights_ib_csv = pd.read_csv("flights.csv")
airlines_ib_csv = pd.read_csv("airlines.csv")

# Connect ibis to pandas
con_ibis = ibis.pandas.connect(
  {
    'flights': flights_ib_csv,
    'airlines': airlines_ib_csv
  }
)

# Checkout the tables
flights_ib = con_ibis.table("flights")
airlines_ib = con_ibis.table("airlines")
```

Non-interactive ibis means that queries are evaluated lazily.

``` python
(
  flights_ib
    .filter(
      (flights_ib.year == 2013) & 
      (flights_ib.month == 1) &
      (flights_ib.arr_delay.notnull())
    )
    .group_by("carrier")
    .aggregate([
      flights_ib["carrier"].count().name("flights"),
      flights_ib["arr_delay"].mean().name("mean_delay")
    ])
)
```

    ##    carrier  flights  mean_delay
    ## 0       9E     1480   10.207432
    ## 1       AA     2724    0.982379
    ## 2       AS       62    8.967742
    ## 3       B6     4413    4.717199
    ## 4       DL     3655   -4.404651
    ## 5       EV     3964   25.160192
    ## 6       F9       59   21.830508
    ## 7       FL      324    3.317901
    ## 8       HA       31   27.483871
    ## 9       MQ     2203    7.883795
    ## 10      OO        1  107.000000
    ## 11      UA     4590    3.175599
    ## 12      US     1554    1.431145
    ## 13      VX      314  -15.280255
    ## 14      WN      985    5.886294
    ## 15      YV       39   13.769231

Building the pipeline in ibis was the most difficult out of the tested libraries. The primary reason is that it was difficult to find help. For that reason I left the ibis pipeline incomplete. The clipping of the `arr_delay` to 0 and the join to `airlines` are missing.

I ran into multiple issues:

-   An error in the predicates argument of `join` when the tables share a column name. Here, it was the column that the join operates on, so I don’t see why an error is raised.
-   Due to the lazy evaluation, is is required to have a `materialize` step after the join.
-   The need to “register” the pandas data frames in ibis rather than operating directly on them as duckdb can
-   I didn’t find documentation indicating how to refer to an other column in a `case` statement

Ibis objects don’t play nice with other Python functions:

``` python
max(flights_pd["arr_delay"]) # 1272.0
# max(flights_ib.arr_delay) # TypeError: 'FloatingColumn' object is not iterable
```

    ## 1272.0

And the [documentation](https://ibis-project.org/docs/user_guide/udf.html) warns:

> UDFs \[User defined functions\] are a complex topic. The UDF API is provisional and subject to change.

Googling and StackOverflow was of little help, only very careful reading of the API docs got me there. Issues like that go away once the user has gained familiarity with the library, but they’ll still remain for new teammates.

The general promise of ibis is amazing: run on everything like SQL, but with the composition and expressiveness of Python. It just seems rough around the edges and with limited help available.

## Conclusion

It’s not a clear-cut choice. None of the options offer a syntax that is as convenient for interactive analysis as dplyr. siuba is the closest to it, but dplyr still has an edge with [tidy evaluation](https://www.tidyverse.org/blog/2019/06/rlang-0-4-0/#a-simpler-interpolation-pattern-with), letting users refer to columns in a data frame by their names (`colname`) directly, without any wrappers. But I’ve also seen it be confusing for newbies to R that mix it up with base R’s syntax. It’s also harder to program with, where it’s necessary to use operators like `{{ }}` and `:=`.

My appreciation for dplyr (and the closely associated tidyr) grew during this research. Not only is it a widely accepted standard like pandas, it can also be used as a translation layer for backends like SQL databases (including duckdb), data.table, and Spark. All while having the most elegant and flexible syntax available.

Personally, I’ll primarily leverage SQL and a OLAP database (such as Clickhouse or Snowflake) running on a server to do the heavy lifting. For steps that are better done locally, I’ll use pandas for maximum compatibility. I find the use of an index inconvenient, but there’s so much online help available on StackOverflow. Github Copilot also deserves a mention for making it easier to pick up. Other use cases can be very different, so I don’t mean to say that my way is the best. For instance, if the data is not already on a server, fast local processing with polars may be best.

Most data science work happens in a team. Choosing a library that all team members are familiar with is critical for collaboration. That is typically SQL, pandas or dplyr. The performance gains from using a less common library like polars have to be weighed against the effort spent learning the syntax as well as the increased likelihood of bugs, when beginners write in a new syntax.

Related articles:

-   [Polars: the fastest DataFrame library you’ve never heard of](https://www.analyticsvidhya.com/blog/2021/06/polars-the-fastest-dataframe-library-youve-never-heard-of/)
-   [What would it take to recreate dplyr in python?](https://mchow.com/posts/2020-02-11-dplyr-in-python/)
-   [Pandas has a hard job (and does it well)](https://mchow.com/posts/pandas-has-a-hard-job/)
-   [dplyr in Python? First impressions of the siuba module](https://bensstats.wordpress.com/2021/09/14/pythonmusings-6-dplyr-in-python-first-impressions-of-the-siuba-小巴-module/)
-   [An Overview of Python’s Datatable package](https://towardsdatascience.com/an-overview-of-pythons-datatable-package-5d3a97394ee9)
-   [Discussion of DuckDB on Hacker News](https://news.ycombinator.com/item?id=24531085)
-   [Discussion of Polars on Hacker News](https://news.ycombinator.com/item?id=29584698)
-   [Practical SQL for Data Analysis](https://hakibenita.com/sql-for-data-analysis)

Photo by <a href="https://unsplash.com/@hharritt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Hunter Harritt</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
