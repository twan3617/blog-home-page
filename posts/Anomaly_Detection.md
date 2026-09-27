---
title:  "Anomaly Detection Project: Completion!"
date:   '2022-03-10'
description: "A research project using streaming data and contextual anomaly detection to identify structural changes in noisy sensor systems."
topics: [Machine Learning, Time Series, Anomaly Detection]
featured: false
---
Recently, I had the pleasure of being part of the team that submitted our results and working prototypes for the "AI for Decision-Making" (Stage 1 Phase 2) project proposed by the [Defence Innovation Network](https://defenceinnovationnetwork.com/), a University-Defence lead iniative to provide funding for projects with relevance to Australian Defence. 

<br>

You can find our work, with demonstrations, in our [repository](https://github.com/sjmluo/Contextually_Situated_Anomaly_Detection). We also wrote up a series of blog posts, aimed at introducing people to our work without getting overly technical. You can find this hosted [here](https://sjmluo.github.io/anomaly/).

<br>

The figures below show how responses from individual sensors can be combined into a steadier signal for anomalous transitions.

<br>

## Figures

<figure>
  <img src="/images/CAC_responses.jpg" alt="Anomaly response curves from individual sensors, with peaks marking possible transitions">
  <figcaption>Each sensor produces an anomaly-response curve; peaks suggest transitions.</figcaption>
</figure>

<figure>
  <img src="/images/combined_CAC_24.jpg" alt="Combined anomaly scores from 24 sensors over a noisy signal">
  <figcaption>Averaging responses from 24 sensors gives a smoother predictor despite noise.</figcaption>
</figure>

<figure>
  <img src="/images/smoothed_CAC_anim.gif" alt="Animated smoothed anomaly score updating as sensor data arrives">
  <figcaption>The score updates as data arrives; red lines mark known transition positions.</figcaption>
</figure>
