---
layout: post
title: Project Checkpoint
---

This document is a checkpoint review of our **Distributed Question Answering System** project for the course **15-418 Parallel Computer Architecture and Programming**, and we will cover the following topics in this report:
* Current Status and Future Schedule
* Summary of Work
* Preliminary Results
* Concerns

<br><br>

## CURRENT STATUS & FUTURE SCHEDULE

The following table shows the current status of our project, and the schedule for the future two weeks segmented into half-week periods.

| Date       | Objectives                                                                                                                                                                      | Status |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| 2017.04.10 | Pre-process Freebase data, split and load into multiple databases. Finalize the Question Answering system structure: entity recognition, query candidate expansion, ranking ... | Done   |
| 2017.04.17 | Train the Bi-diretional LSTM model on the Webquestions dataset in order to compute sentence similarity. Continue working on building the Question Answering system.             | Done   |
| 2017.04.24 | Start working on AMA web service. Distribute computation of each fact candidate to multiple worker nodes.                                                                       |        |
|            | Start working on implementing parallel SVM.                                                                                                                                     |        |
| 2017.05.01 | Finalize implementation of the parallel version of SVM.                                                                                                                         |        |
|            | Combine all components and set up web service.                                                                                                                                  |        |
| 2017.05.08 | Finalize report and presentation. Start working on web service and Question-Answering visualizer if we have time.                                                               |        |

<br><br>


## Summary of Work

For the last two weeks, we've been working on building the entire Question-Answering system from scratch with the following components:
* Entity Linker with [*TagMe*](https://tagme.d4science.org/tagme/)
* Facts Candidates Extraction by sending SPARQL queries to Virtuoso database
* Compute similarity scores with Bi-directional LSTM with pre-trained Embedding (trained pairwisely)
* Feature Engineering
* Rank answer candidates with [*SVM-rank*](https://www.cs.cornell.edu/People/tj/svm_light/svm_rank.html)




#### Deliverables and Goals
I think we have achieved the expected progress according to the schedule posted in the proposal, and we believe we will be able to produce all the deliverables stated in the proposal. Here is a list of goals that we hope to achieve for the final Parallelism competition,
* Robust Question-Answering system with quick response time (within seconds)
* Implement parallel SVM for the ranking phase (new goal)
* Build web server to handle question answering requests elastically (nice to have)
* Build question answering visualizer to display the answering process (nice to have, hopefully on IOS)


#### Plan to Show
We aim to give an interactive demo at the Parallelism competition.

<br><br>


## Preliminary Results

We didn't have any evaluation results yet, since we were still working on parallelizing our Question-Answering system and the SVM ranker. However, we have this short video demo showing what our QA system is capable of doing for now.

<iframe width="800" height="450" src="https://www.youtube.com/embed/wOyso7gFJfU" frameborder="0" allowfullscreen></iframe>

<br><br>


## Concerns

* Schedule: While we have managed to follow the original schedule, we are still a bit behind the project handout due to sequential code implementation, debugging and model training. While we have a working program, we have not fully delved into the parallel implementation, which could bear the weight of a project on its own. 
* Managing workload: Furthermore, the communication-to-computation overhead when dealing with workload distribution to multiple workers is another area that requires our consideration. Both the number of workers spawned and the appropriate amount of work assigned to each worker require extra time to experiment.
* Choice of parallel method: Given the limited remaining time, we are a bit worried about our choice of implementation. There are a bunch of papers that outlines different methods to parallel SVM (e.g. row-based PSVM by Edward Y. Chang et al that both reduces memory requirement in a distributed setting and improves computation time, Parallel Sequential Minimal Optimization(PSMO) illustrated by L. J. Cao et al, that is developed using MPI(We are glad to find that there is MPI support for Python), among others). If time is not allowed, we have to decide our implementation based on theoretical knowledge, rather than implementing all of them and comparing the results.
* Web service: Recently we also started exploring AWS in order to host our service and encountered a little trouble, but we are confident that we could get rid of this problem soon enough.

<br><br>
