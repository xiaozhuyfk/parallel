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
In this project, we built an end-to-end Question Answering system based on Knoledge base (Freebase). 
<br><br>



## BACKGROUND
<br><br>




## APPROACH
#### Fact Candidate Generation
We used the Entity Linker in the Aqqu system [[^fn1]]. During the entity linking process, each subsequence $$s$$ of the query text is matched with all Freebase entities that have name or alias that equal to $$s$$. These Freebase entities are the root entities recognized from the question, and the popolarity score computed with the CrossWikis dataset [[^fn8]] is added to the feature vector in the purpose of measuring entity linking accuracy. The CrossWikis dataset was created by web crawling hyperlinks of Wikipedia entities, and it could be used to measure the empirical distribution over Wikipedia entities. For entities that are not covered by CrossWikis, we only consider the exact name match.

For example, the input question "who inspired obama?" will produce possible subsequences such as 

$$\Big\{ \text{"who inspired"}, \text{"inspired obama"}, \text{"obama"}, ...\Big\}$$

in which the subsequence "obama" match the alias of the entity "Barack Obama" with popularity score of $$0.9029$$.

After identifying the root entities of the question, we can generate the list of fact candidates by extending the root entities with all possible outward edges (relationship) in Freebase. Some example fact candidates generated from the root entity "Barack Obama" are illustrated in the following figure. 

<div style="text-align:center">
<img src="https://xiaozhuyfk.github.io/parallel/images/candidates.png" />
</div>






#### Relation Matching with Bi-directional LSTM
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. The motivation for using LSTM is that LSTM has been proved to be quite effective when solving machine translation problems, and matching relations can be considered as a form of translation.

<div style="text-align:center">
<img src="https://xiaozhuyfk.github.io/parallel/images/LSTM3-chain.png" />
</div>

Each Bi-directional LSTM model is trained pairwisely. Consider there are total of $$n$$ fact candidates for the query $$q$$. We pair them up and get a total of $${n \choose 2} = \frac{n(n-1)}{2}$$ training instances for query $$q$$. For each training instance pair $$(f_1, f_2)$$, the corresponding label will be either $$1$$ or $$-1$$ depending on which fact candidate has higher $F_1$ measure. The $$F_1$$ measure of a fact candidate is computed by the precision and recall of the answers compared with the gold answers provided by the dataset. The pair of fact candidates will then be fed into a pair of identical Bi-directional LSTM model ranking units, and the ranking score of each fact candidate $f$ can later be computed by the ranking unit after the training process.

The query text and the fact candidate components (subject, relation, object) are parsed into word tokens or tri-character-grams as described in [[^fn7]]. For example, the text "barack obama" will be translated into the following tokens,

$$\Big[ \text{#ba, bar, ara, rac, ack, ck#, #ob, oba, bam, ama, ma#}\Big]$$

These tokens will be input to the embedding layer of each model. The structures of the ranking units for both deep learning models are illustrated below.

<div style="text-align:center">
<img src="https://xiaozhuyfk.github.io/parallel/images/lstm.png" />
</div>

In the LSTM Ranking model, the ranking unit will take a single sentence as input consists of the question, subject and relation tokens. With each token input into the embedding matrix, a long list of embedding vectors is passed to the Bidirectional LSTM layer with output size of 1, so that the output of the LSTM layer will be the ranking score of the input fact candidate.

Different from the LSTM Ranking model, the LSTM Joint model will separate the question and relation tokens as separate input instead of combining them together. The corresponding embedding vectors are passed to two separate Bi-directional LSTM layers, and the final ranking score of the fact candidate is the cosine similarity calculated from the output of the two LSTM layers.
<br><br>








#### Distributed Question-Relation Similarity Computation
Given the large amount of triplets to be searched, it provided our group sufficient motivation to distribute the computation to multiple nodes across the network. Each query request can be packaged and sent to a pool of workers, such that data-parallelism can be leveraged upon. In order to achieve this goal, we eventually decided the following structure:

* We adapted a three-level pipelined structure. The ventilator is the producer of the queries and is responsible for packaging the data and sending jobs to workers. The workers are the intermediate consumers of the pipeline and are accountable of the most CPU intensive work. Instead of sending the result back to the producer, the workers will send the result downstream to a common result collector to avoid excessive communication traffic on the producer side. It also reduces contention for physical resources inside the ventilator. If further iterations are required, the message could be sent back from result manager to ventilator in a message batching manner, which will also be illustrated later.

* In order to distribute the data efficiently, we adapted asynchronous socket programming that could be deployed to multiple machines with ease. Given the few dependencies on the queries, as well as the flowing query processing pattern, it makes an asynchronous queue structure preferable. Additionally we used the ventilator to dynamically monitor the queue size of each worker to maintain load-balancing. Requests on socket are scheduled in a round-robin manner out of the concerns of simplicity and unpredictiveness of exact location of the message (ventilator’s buffer, wire, worker’s buffer, to name a few). Neither shared memory nor pipe structure was considered in this scenario, as we are distributing the requests across the network.

* Lastly distributing the data has the additional potential benefit of allowing queries too large to fit into one CPU’s memory to be used, which could be useful when larger knowledge base is incorporated into our application.

<br><br>



## RESULTS
<br><br>



## WORK DISTRIBUTION
Equal work was performed by both project members.
<br><br>



## REFERENCES
[^fn1]: H. Bast and E. Haussmann. More accurate question answering on freebase. In Proceedings of the 24th ACM International on Conference on Information and Knowledge Management, CIKM ’15, pages 1431–1440, New York, NY, USA, 2015. ACM.
[^fn2]: J. Berant, A. Chou, R. Frostig, and P. Liang. Semantic parsing on freebase from question-answer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, EMNLP 2013, 18-21 October 2013, Grand Hyatt Seattle, Seattle, Washington, USA, A meeting of SIGDAT, a Special Interest Group of the ACL, pages 1533– 1544, 2013.
[^fn3]: K. Bollacker, C. Evans, P. Paritosh, T. Sturge, and J. Taylor. Freebase: A collaboratively created graph database for structuring human knowledge. In Proceedings of the 2008 ACM SIGMOD International Conference on Management of Data, SIGMOD ’08, pages 1247– 1250, New York, NY, USA, 2008. ACM.
[^fn4]: T. Joachims. Training linear svms in linear time. In Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’06, pages 217– 226, New York, NY, USA, 2006. ACM.
[^fn5]: Y. Shen, X. He, J. Gao, L. Deng, and G. Mesnil. A latent semantic model with convolutional-pooling structure for information retrieval. In Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management, CIKM ’14, pages 101–110, New York, NY, USA, 2014. ACM.
[^fn6]: Y. Shen, X. He, J. Gao, L. Deng, and G. Mesnil. Learning semantic representations using convolutional neural networks for web search. In Proceedings of the 23rd International Conference on World Wide Web, WWW ’14 Companion, pages 373–374, New York, NY, USA, 2014. ACM.
[^fn7]: W. Yih, M. Chang, X. He, and J. Gao. Semantic parsing via staged query graph generation: Question answering with knowledge base. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing of the Asian Federation of Natural Language Processing, ACL 2015, July 26-31, 2015, Beijing, China, Volume 1: Long Papers, pages 1321–1331, 2015.
[^fn8]: V. I. Spitkovsky and A. X. Chang. A cross-lingual dictionary for english wikipedia concepts. In N. C. C. Chair), K. Choukri, T. Declerck, M. U. Doan, B. Maegaard, J. Mariani, A. Moreno, J. Odijk, and S. Piperidis, editors, Proceedings of the Eight International Conference on Language Resources and Evaluation (LREC’12), Istanbul, Turkey, may 2012. European Language Resources Association (ELRA).

