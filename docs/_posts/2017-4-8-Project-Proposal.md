---
layout: post
title: Project Proposal
---

This document is the proposal of our **Distributed Question Answering System** project for the course **15-418 Parallel Computer Architecture and Programming**, and we will cover the following topics in this proposal:
* Title
* Team
* Summary
* Background
* The Challenge
* Resources
* Goals and Deliverables
* Platform Choice
* Schedule

<br><br>
  
  
## TITLE
Ask Me Anything: Distributed Question Answering System with Freebase

<br><br>

## TEAM
Ziyuan Gong (ziyuang), Hongyu Li (hongyul)

<br><br>

## SUMMARY
We will implement a question answering system trained by deep learning models (LSTM, CDSSM, etc.) that could answer any input questions from users. We will use parallel techniques to speedup the training and ranking phases of the system, and further improve the reponse speed by distributing the computation to multiple workers.

<br><br>


## BACKGROUND
#### What is Freebase?
Freebase is a large collaborative knowledge base containing well structured data that allows machines to access them efficiently. Each entity can be considered as a graph node in the Freebase knowledge graph, and each outward edge together with the connecting entity or attribute represents a piece of fact or knowledge that could be used to answer different kinds of questions.

#### What is Question Answering?
> Question answering (QA) is a computer science discipline within the fields of information retrieval and natural language processing (NLP), which is concerned with building systems that automatically answer questions posed by humans in a natural language. --- Wikipedia

#### What is a typical structure for Question Answering systems?
There are many different approaches to solve the Question Answering problem, such as NLP-based methods, Information Retrieval based methods or Machine Learning techniques, and the system structure may vary. However, our implementation will follow the paradigm below.
* Input: **q**, a question in natural language
* Entity Recognition: identify the root entity **e** in **q**
* Fact Candidate Generation: generate the set of fact candidates **F**, where for each **f** in **F**, **f** is the Freebase triple that is extended from the root entity **e**
* Ranking: rank all possible fact candidates, choose the best one to answer the question **q**

<br><br>

## RESOURCES
We will start from the **aqqu** system provided in the study *More Accurate Question Answering on Freebase* by Hannah Bast and Elmar Haussmann. However, the **aqqu** system was only trained on the **Free917** and the **Webquestions** dataset, so that it does not satisfy our goal that the QA system could answer any input questions. Therefore, we will basically build our own QA system from scratch with similar accuracy but faster response speed.
<br><br>
Papers as references:
* [*More Accurate Question Answering on Freebase*](http://ad-publications.informatik.uni-freiburg.de/CIKM_freebase_qa_BH_2015.pdf)
* [*Semantic Parsing via Staged Query Graph Generation: Question Answering with Knowledge Base*](http://www.aclweb.org/anthology/P15-1128)
* [*Joint Representation Learning of Text and Knowledge for Knowledge Graph Completion*](https://arxiv.org/pdf/1611.04125.pdf)
* [*End-to-end Memory Networks*](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)

<br><br>

## THE CHALLENGE
* Data management: Given the sheer amount of Freebase raw data, it poses a challenge for our team to collect and construct a database to support effective information retrieval. Furthermore, both members of our team have limited prior experience on database, such that it requires extra time for our team to learn new stuff.
* Deep learning model: A significant part of our project relies on keras, a Deep Learning library for Python. Both the adaptation of the framework and the decision for our final deep learning algorithm require experimentation and comparison of various implementations.
* Acceleration: Again, the large amount of data in Freebase provides us with sufficient motivation to adapt parallel mechanism we learnt in class to accelerate the whole process. However, given the hop-by-hop network structure, it poses a challenge for us to find the correct response in a short period of time.

<br><br>

## GOALS AND DELIVERABLES
#### PLAN TO ACHIEVE
One must-achieve for this project is to build a robust Question Answering system. Here is a list of functionalities and requirements that we want our system to achieve:
* Robust, bug-free
* Will give a list of possible answers for any input questions
* Accuracy: correct answer in the top-5 list for 80% of the questions
* Speed: output an answer for any question within a second (the **aqqu** system answers a question for about 5 seconds in average)

#### HOPE TO ACHIEVE
If the project goes really well, we want to achieve the following goals:
* Build a web service to receive user requests
* Build a Question Answering Visualizer that could display the process of finding the answers for the question through Freebase

#### DEMO
* We aim to give an interactive demo during our final presentation. It would be great if we can ask some people to come up with some questions and input them to our Question Answering system, which would make the presentation much more engaging. 
* For evaluation purposes, we will include the speedup graph for multiple approaches in our final report, and also the evaluation results on the **Free917** and **Webquestions** dataset to see whether we have achieved the accuracy objective.

<br><br>

## PLATFORM CHOICE

The system will be implemented in Python, and we will use **Keras** to train the LSTM (Long-short-term-memory) and CDSSM (Covolutional Deep Structured Semantic Model) models.

<br><br>

## SCHEDULE

| Date       | Objectives                                                                                                                                                                      |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2017.04.10 | Pre-process Freebase data, split and load into multiple databases. Finalize the Question Answering system structure: entity recognition, query candidate expansion, ranking ... |
| 2017.04.17 | Train the Bi-diretional LSTM model on the Webquestions dataset in order to compute sentence similarity. Continue working on building the Question Answering system.             |
| 2017.04.24 | Test current system with example question set. Try different parallel techniques to speedup.                                                                                    |
| 2017.05.01 | Launch the Question Answering system on AWS. Try to build the web service to receive Question-Answering request.                                                                |
| 2017.05.08 | Finalize report and presentation. Continue working on web service and Question-Answering visualizer.                                                                            |

<br><br>
