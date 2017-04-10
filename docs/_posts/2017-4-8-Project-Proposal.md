---
layout: post
title: Project Proposal
---

This document is the proposal of our Parallel Question Answering System project for the course **15-418 Parallel Computer Architecture and Programming**, and we will cover the following topics in this proposal:
* Title
* Team
* Summary
* Background
* The Challenge
* Resources
* Goals and Deliverables
* Platform Choice
* Schedule
<br><br><br><br>
  
  
## TITLE
Ask Me Anything: Parallel Question Answering System with Freebase
<br><br><br><br>

## TEAM
Ziyuan Gong (ziyuang), Hongyu Li (hongyul)
<br><br><br><br>

## SUMMARY
We will implement a question answering system trained by deep learning models (LSTM, CDSSM, etc.) that could answer any input questions from users. We will use parallel techniques to speedup the training and ranking phases of the system, and further improve the reponse speed by distributing the computation to multiple workers.
<br><br><br><br>


## BACKGROUND
Freebase is a large collaborative knowledge base containing well structured data that allows machines to access them efficiently. Each entity can be considered as a graph node in the Freebase knowledge graph, and each outward edge together with the connecting entity or attribute represents a piece of fact or knowledge that could be used to answer different kinds of questions.
<br><br><br><br>

## RESOURCES
We will start from the **aqqu** system provided in the study *More Accurate Question Answering on Freebase* by Hannah Bast and Elmar Haussmann. However, the **aqqu** system was only trained on the **Free917** and the **Webquestions** dataset, so that it does not satisfy our goal that the QA system could answer any input questions. Therefore, we will basically build our own QA system from scratch with similar accuracy but faster response speed.
<br>
Papers as references:
* [*More Accurate Question Answering on Freebase*](http://ad-publications.informatik.uni-freiburg.de/CIKM_freebase_qa_BH_2015.pdf)
* [*Semantic Parsing via Staged Query Graph Generation: Question Answering with Knowledge Base*](http://www.aclweb.org/anthology/P15-1128)
* [*Joint Representation Learning of Text and Knowledge for Knowledge Graph Completion*](https://arxiv.org/pdf/1611.04125.pdf)
* [*End-to-end Memory Networks*](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)

<br><br><br><br>

## THE CHALLENGE
<br><br><br><br>

## GOALS AND DELIVERABLES
#### PLAN TO ACHIEVE

#### HOPE TO ACHIEVE

#### DEMO
<br><br><br><br>

## PLATFORM CHOICE
<br><br><br><br>

## SCHEDULE
<br><br><br><br>
