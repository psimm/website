---
title: "Human and AI Decision Making in a Game of Innovation and Imitation"
subtitle: ""
excerpt: "Thesis research project at Aalborg University. It involved an implementation of DeepMind's Alpha Zero in R and a business game built in R Shiny."
date: 2018-02-01
author: "Paul Simmering"
draft: false
tags:
  - Game
categories:
  - Shiny
  - R
  - Machine learning
layout: single
links:
- icon: file-alt
  icon_pack: fas
  name: Master thesis
  url: https://projekter.aau.dk/projekter/da/studentthesis/human-and-ai-decision-making-in-a-game-of-innovation-and-imitation(9121a1ed-d5d7-4cf0-b725-41f822533544).html
- icon: slideshare
  icon_pack: fab
  name: Presentation
  url: /talk/researchplus2018/researchplus2018.pdf
- icon: gamepad
  icon_pack: fas
  name: Shinyapp Game
  url: https://psimm.shinyapps.io/business_game
- icon: github
  icon_pack: fab
  name: Github
  url: https://psim.shinyapps.io/business_game/
---

My master thesis investigates the use of artificial intelligence (AI) in managerial decision making. The thesis was supervised by Assoc. Prof. Daniel S. Hain and Assoc. Prof. Roman Jurowetzki at Aalborg University.

Current AI is narrow, specialized for single tasks and cannot be applied to others. However, recent developments in general game-playing algorithms suggest that AI will become more generally applicable. In managerial decision making, it could be used as a decision support systems or as an autonomous decision maker. This idea of an artificial business decision maker is studied along four research questions.

1. How do AI and human thought processes differ?
2. Do AIs and humans make qualitatively different business decisions?
3. What are the dynamics of competition and cooperation between humans and AI? 
4. Are there potential problems in value alignment between a business and its AI?

![](/project/masterthesis/howtoplay.png)

The study approaches these questions by example of a business game. The game depicts competition between two firms in a consumer goods market and emphasizes innovation and imitation strategies in product development, as well as vertical and horizontal product differentiation. It is played by an AI and human participants. The agent combines Monte Carlo Tree Search with prediction of outcomes using an artificial neural network. Six human participants played two games each against that agent. While playing, they gave a think-aloud protocol. The research questions are answered by combining insights from a content analysis of the protocols and an analysis of the AI’s architecture and processes.

The AI combines forward reasoning using tree search and evaluation of situations with artificial neural networks. This parallels humans’ thought processes that combine conscious, effortful thinking with unconscious, effortless evaluation. The differences lie in AI’s superior computational abilities, humans’ superior ability to learn from small samples and humans’ conscious and unconscious social behavior and emotions. The absence of this social behavior causes AI to act qualitatively differently -- to consider actions that humans do not. This divergence can take the form of breaches of norms of reciprocity and unorthodox pursuit of a utility function. Instructing an AI is difficult, because humans have utility functions with many inputs that have complex relationships among each other, and may be unaware of elements until they come to bear. Value alignment is an on- going challenge for businesses and policy makers. Further, firms have to learn how to best incorporate AI in their decision making. This includes training employees in the use of AI assistants, developing transparent algorithms and developing an awareness for situations in which the use of AI is inappropriate for technical, legal or social reasons.
