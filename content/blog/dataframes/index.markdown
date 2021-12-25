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

I’m a long time R user and lately I’ve seen more and more signals that it’s worth investing into learning more Python. I use it for NLP with [spaCy](https://spacy.io) and to build functions on [AWS Lambda](https://aws.amazon.com/lambda/features/) (though the recent [lambdr](https://mdneuzerling.com/post/serverless-on-demand-parametrised-r-markdown-reports-with-aws-lambda/) package gave R some great tools there too). Further, there are many more data API libraries available for Python than for R.

Michael Chow, developer of [siuba](https://github.com/machow/siuba), a Python port of dplyr on top of pandas [wrote](https://mchow.com/posts/pandas-has-a-hard-job/):

> It seems like there’s been a lot of frustration surfacing on twitter lately from people coming from R—especially if they’ve used dplyr and ggplot—towards pandas and matplotlib. I can relate. I’m developing a port of dplyr to python. But in the end, it’s probably helpful to view these libraries as foundational to a lot of other, higher-level libraries (some of which will hopefully get things right for you!).

And that summarizes how I feel about it: The syntax of pandas feels inferior to dplyr and I wish I could have the same fluency as with dplyr. The higher-level libraries he mentions come with a problem though: There’s no accepted standard. So investing the time to learn one of them may be futile as the next project one is working on could be with other developers who are fluent in a different one.

## The contenders

Python has a larger package ecosystem and ways of doing things than R, which seems more centralized thanks to CRAN. Adopting Python means making many choices on which libraries to invest time into learning. The first thing I needed was a way to manipulate data frames. I’ve narrowed it down to 3 choices:

-   [Pandas](https://pandas.pydata.org): The most commonly used library and the one with the most tutorials and Stack Overflow answers available.
-   [siuba](https://github.com/machow/siuba): A port of dplyr to Python, built on top of pandas
-   [Polars](https://www.pola.rs): The fastest library available. It’s a new library that doesn’t have nearly as many users or help available. But according to the [H2Oai ops benchmark](https://h2oai.github.io/db-benchmark/), it runs 3-10x faster than Pandas.
-   [Duckdb](https://www.pola.rs): Use an in-memory OLAP database instead of a dataframe. I know SQL, so this is the easiest one to pick up.

There are far more options and this is my shortlist. I have excluded the other options in the benchmark for these reasons:

-   Slower than polars (dask, Arrow, Modin)
-   Not mature enough, as shown by lower activity on Github (pydatatable)
-   Requires other software or a server (ClickHouse)
-   Not in Python (DataFrames.jl)
-   Meant for GPU only (cuDF)
-   Wrappers like [ibis](https://ibis-project.org/docs/index.html) that delegate computation to pandas or a server running SQL. Technically siuba is also in this group, but as I’m coming from dplyr I had to include it.

The benchmark provides a comparison of performance, but another important factor is popularity and maturity. A more mature library has a more stable API, better test coverage and there is more help available online, such as on StackOverflow.

``` r
library(ggplot2)
libs <- data.frame(
  library = c("pandas", "siuba", "polars", "duckdb", "dplyr", "data.table"),
  language = c("Python", "Python", "Python", "SQL", "R", "R"),
  stars = c(32100, 732, 3900, 4100, 3900, 2900)
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

Github stars are not a perfect proxy. For instance, dplyr is much more mature than its star count suggests. Comparing the completen and completeness of the documentation of dplyr and polars reveals that it’s a night and day difference.

My reference is my current use of [dplyr](https://dplyr.tidyverse.org) in R. When I need more performance, I use [tidytable](https://github.com/markfairbanks/tidytable) to get the speed of data.table with the grammar of dplyr. I also use [dbplyr](https://dbplyr.tidyverse.org) a lot, which translates dplyr to SQL. It’s composeable, which actually makes it superior to SQL for me in most use cases.

With that out of the way, here’s a heavily biased comparison of the four Python packages.

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
flights |>
  filter(year == 2013, month == 1, !is.na(arr_delay)) |> 
  mutate(arr_delay = replace(arr_delay, arr_delay < 0, 0)) |>
  left_join(airlines, by = "carrier") |>
  group_by(airline = name) |>
  summarise(flights = n(), mean_delay = mean(arr_delay)) %>% 
  arrange(desc(mean_delay))
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
```

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
flights_pd = pd.read_csv("flights.csv")
airlines_pd = pd.read_csv("airlines.csv")
```

`pandas.read_csv` read the header and conveniently inferred the column types.

``` python
(flights_pd
  .query("year == 2013 & month == 1 & arr_delay.notnull()")
  .assign(arr_delay = flights_pd.arr_delay.clip(lower = 0))
  .merge(airlines_pd, how = "left", on = "carrier")
  .rename(columns = {"name": "airline"})
  .groupby("airline")
  .agg(flights = ("airline", "count"), mean_delay = ("arr_delay", "mean"))
  .sort_values(by = "mean_delay", ascending = False))
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
```

Rows with missing values for `arr_delay` are dropped implicitly in the `agg` step.

Pandas uses a row index, which is basically a special column. Base R also has this with row names, though the tidyverse and tibbles have largely removed them from common use.

Pandas has the widest API, offering hundreds of functions for every conceivable manipulation.

## Polars: Fastest

Polars is written in Rust and also offers a Python API. It comes in two flavors: eager and lazy. Lazy evaluation is similar to how dbplyr and dtplyer work: until asked, nothing is evaluated. This enables performance gains by reordering the commands being executed. But it’s a little less convenient for interactive analysis. I’ll use the eager API here.

``` python
import polars as pl

# Import from CSV
flights_pl = pl.read_csv("flights.csv")
airlines_pl = pl.read_csv("airlines.csv")
```

The API is leaner than pandas, requiring to memorize fewer functions and patterns. Though this can also be seen as less feature-complete. Pandas, for example has a dedicated `clip` function.

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

There isn’t nearly as much help available for problems with polars as for with pandas. Often, I had to use trial and error based on the documentation. While the documentation is good, it can’t answer every question.

A comparison of polars and pandas is available in the [polars documentation](https://pola-rs.github.io/polars-book/user-guide/coming_from_pandas.html?highlight=assign#column-assignment). A notable difference is that polars doesn’t have a concept of indexes, just like tibbles in R.

## DuckDB: Highly compatible and easy for SQL users

``` python
import duckdb

con = duckdb.connect(database = ':memory:')

# Import from CSV
con.execute(
  "CREATE TABLE 'flights' AS "
  "SELECT * FROM read_csv_auto('flights.csv', header = True);"
  "CREATE TABLE 'airlines' AS "
  "SELECT * FROM read_csv_auto('airlines.csv', header = True);"
)
## <duckdb.DuckDBPyConnection object at 0x11aa052b0>
```

``` python
con.execute(
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
```

The performance is closer to polars than to pandas.

A big plus is the ability to handle larger than memory data.

DuckDB can also operate directly on a pandas dataframe.

The code is portable to R, C, C++, Java and other programming languages the duckdb has [APIs](https://duckdb.org/docs/api/overview). It’s also portable when the logic is taken to a DB like [Postgres](https://www.postgresql.org), or [Snowflake](https://www.snowflake.com/), or is ported to an ETL framework like [DBT](https://github.com/dbt-labs/dbt-core).

This stands in contrast to polars and pandas code, which has to be rewritten from scratch. It also means that the skill gained in manipulating data translates well to other situations - SQL has been around for more than 40 years. Something that can’t be said about any Python library. Learning SQL is future-proofing ones career.

While these are big plusses, duckdb isn’t as convenient as Polars and Pandas for interactive data exploration. The SQL isn’t as composeable.

Composing SQL queries requires many common table expressions (CTEs, `WITH x AS (SELECT ...)`).

Plus, writing strings rather than actual Python is awkward and many editors don’t provide syntax highlighting within the strings (Jetbrains editors like [PyCharm](https://www.jetbrains.com/pycharm/) and [DataSpell](https://www.jetbrains.com/dataspell/) do).

SQL is less expressive than Python, especially when the names of output columns are unknown.

It lacks shorthands. It’s also harder to write custom functions in SQL. With pandas and polars, custom operations are just one lambda away.

Using duckdb without pandas doesn’t seem feasible for exploratory data analysis, because graphing packages like seaborn and plotly expect a pandas data frame or similar as an input.

Speed of iteration is critical: the faster one can iterate, the more hypotheses about the data can be tested.

## Conclusion

It’s not a clear-cut choice. Each seems more useful in it’s own arena.

None of the three options offer a syntax that is as convenient for interactive analysis as dplyr. Polars is the closest to it, but dplyr still has an edge with [tidy evaluation](https://www.tidyverse.org/blog/2019/06/rlang-0-4-0/#a-simpler-interpolation-pattern-with), letting users refer to columns in a data frame by their names (`colname`) rather than as strings `"colname"`or constructs like `pl.col("colname")`. While this is nice for quickly writing code, I’ve also seen it be confusing for newbies to R that mix it up with base R’s syntax. It’s also harder to program with, where it’s necessary to use operators like `{{ }}` and `:=`.

Personally, I’ll leverage my existing knowledge and rely on SQL and an OLAP database (such as Snowflake) to do the heavy lifting. For steps that are better done locally, I’ll use pandas for maximum compatibility. The syntax isn’t my favorite, but there’s so much online help available that StackOverflow has the answer for almost any problem. Github Copilot also deserves a mention for making it easier to pick up.

Most data science work happens in a team. Choosing a library that all team members are familiar with is critical for collaboration. That is typically SQL, pandas or dplyr. The performance gains from using a less common library like polars have to be weighed against the effort spent learning the syntax as well as the increased likelihood of bugs, when beginners write in a new syntax.

Related articles:

-   [Polars: the fastest DataFrame library you’ve never heard of](https://www.analyticsvidhya.com/blog/2021/06/polars-the-fastest-dataframe-library-youve-never-heard-of/)
-   [What would it take to recreate dplyr in python?](https://mchow.com/posts/2020-02-11-dplyr-in-python/)
-   [Pandas has a hard job (and does it well)](https://mchow.com/posts/pandas-has-a-hard-job/)

Photo by <a href="https://unsplash.com/@hharritt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Hunter Harritt</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
