---
layout: post
title: Final Writeup
---

This document is a final report of our **Distributed Question Answering System** project for the course **15-418 Parallel Computer Architecture and Programming**, 
and we will cover the following topics in this report:
* Summary
* Background
* Approach
* Results

<br><br>

## SUMMARY
<br><br>



## BACKGROUND
<br><br>




## APPROACH
#### Fact Candidate Generation



#### Relation Matching with Bi-directional LSTM



#### Distributed Question-Relation Similarity Computation
Given the large amount of triplets to be searched, it provided our group sufficient motivation to distribute the computation to multiple nodes across the network. Each query request can be packaged and sent to a pool of workers, such that data-parallelism can be leveraged upon. In order to achieve this goal, we eventually decided the following structure:

* We adapted a three-level pipelined structure. The ventilator is the producer of the queries and is responsible for packaging the data and sending jobs to workers. The workers are the intermediate consumers of the pipeline and are accountable of the most CPU intensive work. Instead of sending the result back to the producer, the workers will send the result downstream to a common result collector to avoid excessive communication traffic on the producer side. It also reduces contention for physical resources inside the ventilator. If further iterations are required, the message could be sent back from result manager to ventilator in a message batching manner, which will also be illustrated later.

* In order to distribute the data efficiently, we adapted asynchronous socket programming that could be deployed to multiple machines with ease. Given the few dependencies on the queries, as well as the flowing query processing pattern, it makes an asynchronous queue structure preferable. Additionally we used the ventilator to dynamically monitor the queue size of each worker to maintain load-balancing. Requests on socket are scheduled in a round-robin manner out of the concerns of simplicity and unpredictiveness of exact location of the message (ventilator’s buffer, wire, worker’s buffer, to name a few). Neither shared memory nor pipe structure was considered in this scenario, as we are distributing the requests across the network.

* Lastly distributing the data has the additional potential benefit of allowing queries too large to fit into one CPU’s memory to be used, which could be useful when larger knowledge base is incorporated into our application.


#### Learning-to-Rank

<br><br>



## RESULTS
<br><br>



## REFERENCES
<br><br>

## WORK DISTRIBUTION
Equal work was performed by both project members.
<br><br>

