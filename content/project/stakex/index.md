---
title: "StakeX - Organizational Networks from Web Research"
subtitle: "Network analysis approach and two R Shiny apps"
excerpt: "Shiny apps for data entry and network analysis of organizational networks with a geospatial component."
date: 2020-09-30
author: "Paul Simmering"
draft: false
tags:
  - Visualization
categories:
  - Shiny
  - R
  - Network analysis
layout: single
links:
- icon: door-open
  icon_pack: fas
  name: demo
  url: https://teamqapp.de/shiny/stakex_operativ/
- icon: slideshare
  icon_pack: fab
  name: presentation
  url: /talk/comes2020/
---

![](/project/stakex/stakex1.png)

StakeX is a network analysis approach  for public relations projects. It is in use at Q Insight Agency with clients in public transportation and the energy sector. The approach consists of:

- methods for gathering public data on relevant stakeholders
- building a network based on sociological theory
- analyzing the network through use of force-based layout algorithms and centrality statistics
- extracting valuable insights for customers

To learn more about the method, see the [talk](/talk/comes2020/) by Thomas Perry and me or check the methods section of the demo app.

I led the development of two Shiny apps for this project. The first is a CRUD app that facilitates data entry into a PostgreSQL database and ensures data integrity. The second is a platform for interactive data analysis featuring graphs with [visNetwork](https://datastorm-open.github.io/visNetwork/) and maps with [leaflet](https://rstudio.github.io/leaflet/). Both are hosted on EC2 instances on AWS.

Tech stack: R, PostgreSQL, AWS EC2

![](/project/stakex/stakex2.png)
![](/project/stakex/stakex3.png)
