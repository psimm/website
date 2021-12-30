---
title: "Global Patent Explorer"
subtitle: "Award-winning Shiny App for patent statistics"
excerpt: "Award-winning Shiny app for display of worldwide patent statistics and citation patterns. Freelance work for Aalborg University."
date: 2018-07-01
author: "Paul Simmering"
draft: false
tags:
  - data-visualization
categories:
  - R
  - Shiny
  - Network-analysis
# layout options: single or single-sidebar
layout: single
links:
- icon: door-open
  icon_pack: fas
  name: Shiny app
  url: https://psim.shinyapps.io/patent
- icon: github
  icon_pack: fab
  name: Github
  url: https://github.com/psimm/patent_vis
- icon: slideshare
  icon_pack: fab
  name: presentation
  url: /talk/gor2019
---

Freeelance work for Aalborg University, commissioned by Assoc. Prof. Daniel S. Hain and Assoc. Prof. Roman Jurowetzki. It visualizes the results of a paper on natural language processing on patent texts:

> D.S. Hain, R. Jurowetzki, T. Buchmann, P. Wolf (2018), A Vector Worth a Thousand Counts: A Temporal Semantic Similarity Approach to Patent Impact Prediction. Available at [http://vbn.aau.dk/en/publications/a-vector-worth-a-thousand-counts(855d9758-d017-4b4a-baf5-8b7e72a1c223).html](http://vbn.aau.dk/en/publications/a-vector-worth-a-thousand-counts(855d9758-d017-4b4a-baf5-8b7e72a1c223).html)

![](/project/gpexp/gpexp.png)

This tool lets users map and visualize inventive and innovative activity around the globe. The explorer relies on a series of novel indicators that combine insights from large-scale natural language processing and established patent analysis techniques and provide insights about dimensions such as technological originality or future orientation. Users can explore the dataset on country or city level, select time-ranges and technologies. The app features rich visualizations including a world map, network plots that show relations between countries and cities, and customizable statistical plots.

The app is a winner of the first IPSDM (Intellectual property statistics for descision makers) “Big Data Analytics” Challenge (2018) by the European Union Intellectual Property Office.

![](/project/gpexp/euipo.png)

Tech stack: R, Shiny
