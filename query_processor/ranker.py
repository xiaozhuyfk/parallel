from __future__ import print_function

import logging
import globals
import sys
import os
import modules
from util import (
    codecsWriteFile,
    codecsReadFile,
    codecsDumpJson,
    codecsLoadJson,
    computeF1
)
import subprocess
import re
from evaluation import load_eval_queries
import numpy as np
from model import (
    LSTMPointwise,
    LSTMPairwise,
    LSTMJointPairwise,
    DSSMPairwise,
    EmbeddingJointPairwise,
    vectorize_sentence_one_hot
)
import wikipedia
import nltk.data

logger = logging.getLogger(__name__)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def tokenize_term(t):
    return re.sub('[?!@#$%^&*,()_+=\'\d\./;]', '', t).lower()


class FeatureVector(object):

    def __init__(self, query, relevance, candidate):
        self.query = query
        self.query_id = query.id
        self.candidate = candidate
        self.relevance = relevance
        self.f1 = candidate.f1
        self.features = {}
        self.format = "%d qid:%s %s# %s\n"

    def add(self, i, value):
        self.features[i] = value

    def get(self, i):
        return self.features.get(i)

    def update(self, i, value):
        self.features[i] = value

    def __iter__(self):
        for i in self.features:
            yield i

    def __str__(self):
        indicator = "" #self.candidate.subject + " " + self.candidate.relation
        vec = ""
        for i in self.features:
            vec += str(i) + ":" + str(self.features[i]) + " "
        return self.format % ((self.f1 >= 0.5) * 1, str(self.query_id), vec, indicator)

    def __len__(self):
        return len(self.features)


def ngramize(tokens, n):
    assert(n > 0)
    ngrams = []
    for token in tokens:
        if token:
            t = "#" * (n-1) + token + "#" * (n-1)
            for idx in xrange(len(token) + n - 1):
                ngrams.append(t[idx:idx+n])
    return ngrams

class FactCandidate(object):
    def __init__(self, config_options, query, subject, sid, score, relation, response):
        self.config_options = config_options
        self.query = query
        self.question = query.utterance[:-1]
        self.answers = query.target_result

        self.subject = subject
        self.sid = sid
        self.score = score
        self.relation = relation
        self.response = response

        self.objects = response["objects"]
        self.oid = response["oid"]
        self.feature_idx = 1

        # word based
        self.query_tokens = [tokenize_term(t) for t in self.question.split()]
        self.subject_tokens = [re.sub('[?!@#$%^&*,()_+=\'/]', '', t).lower()
                               for t in subject.split()]
        relations = re.split("\.\.|\.", self.relation.split("\n")[-1])[-2:]
        #[re.split("\.\.|\.", r) for r in self.relation.split("\n")]
        self.relation_tokens = [tokenize_term(e)
                                for t in relations
                                for e in re.split("\.\.|\.|_", t)]

        # character tri-gram
        self.query_trigram = ngramize(self.query_tokens, 3)
        self.subject_trigram = ngramize(self.subject_tokens, 3)
        self.relation_trigram = ngramize(self.relation_tokens, 3)

        self.relevance = 0
        for object in self.objects:
            if object in self.answers:
                self.relevance = 1
                break

        self.sentence = self.query_tokens + self.subject_tokens + self.relation_tokens
        self.sentence_size = len(self.sentence)
        self.sentence_trigram = self.query_trigram + self.subject_trigram + \
                                self.relation_trigram
        self.sentence_trigram_size = len(self.sentence_trigram)
        self.candidate_sentence = self.subject_tokens + self.relation_tokens
        self.candidate_sentence_trigram = self.subject_trigram + self.relation_trigram

        self.vocab = set(self.sentence)
        self.vocab_trigram = set(self.sentence_trigram)
        self.entity_vocab = set(self.subject_tokens)

        self.f1 = computeF1(self.answers, self.objects)[2]

        self.support = response["support"]

        """
        self.top_sentence_with_question = []
        self.top_sentence_with_question_trigram = []
        self.top_sentence_with_candidate = []
        self.top_sentence_with_candidate_trigram = []
        question_set = set(self.query_tokens)
        question_trigram_set = set(self.query_trigram)
        candidate_set = set(self.subject_tokens + self.relation_tokens)
        candidate_trigram_set = set(self.subject_trigram + self.relation_trigram)

        max_question_overlap = 0
        max_question_trigram_overlap = 0
        max_candidate_overlap = 0
        max_candidate_trigram_overlap = 0
        for sentence in self.support:
            sentence_tokens = [tokenize_term(t) for t in sentence.split()]
            sentence_trigram_tokens = ngramize(sentence_tokens, 3)
            if len(set(sentence_tokens) & question_set) > max_question_overlap:
                max_question_overlap = len(set(sentence_tokens) & question_set)
                self.top_sentence_with_question = sentence_tokens
            if len(set(sentence_trigram_tokens) & question_trigram_set) > max_question_trigram_overlap:
                max_question_trigram_overlap = len(set(sentence_trigram_tokens) & question_trigram_set)
                self.top_sentence_with_question_trigram = sentence_trigram_tokens
            if len(set(sentence_tokens) & candidate_set) > max_candidate_overlap:
                max_candidate_overlap = len(set(sentence_tokens) & candidate_set)
                self.top_sentence_with_candidate = sentence_tokens
            if len(set(sentence_trigram_tokens) & candidate_trigram_set) > max_candidate_trigram_overlap:
                max_candidate_trigram_overlap = len(set(sentence_trigram_tokens) & candidate_trigram_set)
                self.top_sentence_with_candidate_trigram = sentence_trigram_tokens

        self.top_sentence_with_question = self.top_sentence_with_question[:28]
        self.top_sentence_with_candidate = self.top_sentence_with_candidate[:28]
        self.top_sentence_with_question_trigram = self.top_sentence_with_question_trigram[:203]
        self.top_sentence_with_candidate_trigram = self.top_sentence_with_candidate_trigram[:203]
        self.max_question_overlap = max_question_overlap
        self.max_question_trigram_overlap = max_question_trigram_overlap
        self.max_candidate_overlap = max_candidate_overlap
        self.max_candidate_trigram_overlap = max_candidate_trigram_overlap
        """

        """
        if (len(self.support) == 0):
            self.support = set([])
            for o in self.oid[:5]:
                self.support |= set(modules.support_sentence_extractor.get_support_sentence_with_pair(self.sid, o))
            self.support = list(self.support)
        """

        graph_tokens = [" ".join(self.subject_tokens),
                        " ".join(self.relation_tokens),
                        str(self.objects[:5]).encode("utf-8")]
        self.graph_str = " --> ".join(graph_tokens)

        """
        # support sentences
        sentences = modules.wiki_extractor.extract_wiki_page(
            query.dataset,
            query,
            self.subject,
            sid.replace(".", "-")
        )
        self.support = set([])
        for object in self.objects:
            object = object.lower()
            for sent in sentences:
                sent = sent.lower()
                if self.subject.lower() in sent and object in sent:
                    self.support.add(sent)

        # support sentences from wikipedia summary
        sentences = []
        try:
            text = wikipedia.summary(self.subject).lower()
            paragraphs = text.strip().split("\n")
            sentences = [tokenizer.tokenize(p) for p in paragraphs if p]
            sentences = [s for p in sentences for s in p]
        except:
            pass
        self.support_summary = set([])

        for object in self.objects:
            object = object.lower()
            for sent in sentences:
                sent = sent.lower()
                if self.subject.lower() in sent and object in sent:
                    self.support_summary.add(sent)
        """

        """
        graph_tokens = [" ".join(self.subject_tokens),
                        " ".join(self.relation_tokens),
                        str(self.objects[:5])]
        graph_str = " --> ".join(graph_tokens)
        self.message = "Entity Score = %f, F1 = %f, graph = %s\n" % (self.score, self.f1, graph_str)
        self.message += "Number of support sentences = %d\n" % (len(self.support))
        self.message += "Example support sentence:\n"
        if len(self.support) > 0:
            for sent in list(self.support)[:4]:
                self.message += sent + "\n"
        """

    def top_sentence_score(self, model):
        self.top_sentence = "EMPTY"
        if len(self.support) == 0: return 0
        sentence_tokens = [[tokenize_term(t) for t in re.split('[\[\]\s|\']', sentence) if t][:28] for sentence in self.support]
        predictions = model.predict_with_sent(self.query_tokens, sentence_tokens, 28).flatten()
        idx = np.argmax(predictions)
        self.top_sentence = self.support[idx]
        return predictions[idx]

    def support_sentence_score(self, model):
        if len(self.support) == 0: return 0
        sentence_tokens = [[tokenize_term(t) for t in sentence.split()][:28] for sentence in self.support]
        predictions = model.predict_with_sent(self.query_tokens, sentence_tokens, 28).flatten()
        return np.sum(predictions) / len(predictions)



    def get_support_sentence(self):
        self.support = set([])
        for o in self.oid[:5]:
            self.support |= set(modules.support_sentence_extractor.get_support_sentence_with_pair(self.sid, o))

    def __str__(self):
        graph_tokens = [" ".join(self.subject_tokens),
                        " ".join(self.relation_tokens),
                        str(self.objects[:5])]
        graph_str = " --> ".join(graph_tokens)
        self.message = "Entity Score = %f, F1 = %f, graph = %s\n" % (self.score, self.f1, graph_str)
        self.message += "Number of support sentences = %d\n" % (len(self.support))
        #self.message += "Example support sentence:\n"
        #if len(self.support) > 0:
        #    for sent in list(self.support)[:4]:
        #        self.message += sent.encode("utf-8") + "\n"
        return self.message

    def vectorize_sentence(self, word_idx, sentence, sentence_size):
        sentence_idx = [word_idx.get(t, 0) for t in sentence] + \
                        (sentence_size - len(sentence)) * [0]
        return sentence_idx

    def update_feature(self, idx, value):
        self.feature_vector.add(idx, value)

    def add_feature(self, value):
        self.feature_vector.add(self.feature_idx, value)
        self.feature_idx += 1

    def extract_features(self):
        relevance = 0
        for object in self.objects:
            if object in self.answers:
                relevance = 1

        vector = FeatureVector(self.query, relevance, self)
        self.feature_vector = vector

        # Add entity linking score
        self.add_feature(float(self.score))

        # Add wiki popularity
        self.add_feature(len(self.support))

        # term overlap with question
        # self.add_feature(self.max_question_overlap)
        self.add_feature(0)

        # term overlap with candidate
        # self.add_feature(self.max_candidate_overlap)

        # Add wiki summary popularity
        #self.add_feature(len(self.support_summary))

        # Add number of nodes
        # relations = self.relation.split("\n")
        # vector.add(2, float(len(relations) + 1))

        # Add number of answers
        # vector.add(2, float(len(self.objects)))

        #self.feature_vector = vector
        return self.feature_vector




class Ranker(object):

    def __init__(self,
                 config_options,
                 svmRankParamC,
                 svmRankLearnPath,
                 svmRankClassifyPath,
                 svmRankModelFile,
                 svmTrainingFeatureVectorsFile,
                 svmTestingFeatureVectorsFile,
                 svmFactCandidateScores):
        self.config_options = config_options
        self.svmRankParamC = svmRankParamC
        self.svmRankLearnPath = svmRankLearnPath
        self.svmRankClassifyPath = svmRankClassifyPath
        self.svmRankModelFile = svmRankModelFile
        self.svmTrainingFeatureVectorsFile = svmTrainingFeatureVectorsFile
        self.svmTestingFeatureVectorsFile = svmTestingFeatureVectorsFile
        self.svmFactCandidateScores = svmFactCandidateScores

    @staticmethod
    def init_from_config(config_options):
        svmRankParamC = config_options.get('SVM', 'paramc')
        svmRankLearnPath = config_options.get('SVM', 'learn-path')
        svmRankClassifyPath = config_options.get('SVM', 'classify-path')
        svmRankModelFile = config_options.get('SVM', 'rank-model-file')
        svmTrainingFeatureVectorsFile = config_options.get('SVM', 'training-vector-file')
        svmTestingFeatureVectorsFile = config_options.get('SVM', 'testing-vector-file')
        svmFactCandidateScores = config_options.get('SVM', 'testing-rank-scores')
        return Ranker(config_options,
                      svmRankParamC,
                      svmRankLearnPath,
                      svmRankClassifyPath,
                      svmRankModelFile,
                      svmTrainingFeatureVectorsFile,
                      svmTestingFeatureVectorsFile,
                      svmFactCandidateScores)

    def svm_learn(self):
        logger.info("Start SVM Training ...")
        cmd = [self.svmRankLearnPath,
               "-c",
               self.svmRankParamC,
               self.svmTrainingFeatureVectorsFile,
               self.svmRankModelFile]
        p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        p.wait()

    def svm_rank(self):
        #logger.info("Start SVM Ranking ...")
        cmd = [self.svmRankClassifyPath,
               self.svmTestingFeatureVectorsFile,
               self.svmRankModelFile,
               self.svmFactCandidateScores]
        p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        p.wait()

    def nomalize_features(self, candidates):
        if candidates == []:
            return []
        length = len(candidates[0].feature_vector)

        minimums = np.array([float("inf")] * length)
        maximums = np.array([-float("inf")] * length)
        for candidate in candidates:
            vec = candidate.feature_vector
            for i in vec:
                if vec.get(i) < minimums[i-1]:
                    minimums[i-1] = vec.get(i)
                if vec.get(i) > maximums[i-1]:
                    maximums[i-1] = vec.get(i)
        for candidate in candidates:
            vec = candidate.feature_vector
            for i in vec:
                if maximums[i-1] == minimums[i-1]:
                    new = 0.0
                else:
                    new = (vec.get(i) - minimums[i-1]) / float((maximums[i-1] - minimums[i-1]))
                vec.update(i, new)
            candidate.feature_vector = vec
        return candidates

    def extract_candidates_with_f1(self, dataset):
        queries = load_eval_queries(dataset)
        correct = []
        positive = []
        negative = []
        for query in queries:
            print("Processing query " + str(query.id) + " " * 10 + "\r", end="")
            json = modules.extractor.extract_fact_list_with_entity_linker(dataset, query)
            facts = json["facts"]
            for ie in facts:
                subject = ie["subject"]
                sid = ie["sid"]
                score = ie["score"]
                relations = ie["relations"]
                for rel in relations:
                    fact_candiate = FactCandidate(self.config_options,
                                                  query,
                                                  subject,
                                                  sid,
                                                  score,
                                                  rel,
                                                  relations[rel])
                    if fact_candiate.f1 == 1.0:
                        correct.append(fact_candiate)
                    if fact_candiate.f1 > 0:
                        positive.append(fact_candiate)
                    else:
                        negative.append(fact_candiate)
        return correct, positive, negative

    def extract_wiki_data_with_index(self, dataset, idx):
        queries = load_eval_queries(dataset)
        query = queries[idx]
        logger.info("Processing query " + str(query.id))
        json = modules.extractor.extract_fact_list_with_entity_linker(dataset, query)
        facts = json["facts"]
        for ie in facts:
            logger.info("Processing root entity: " + ie["subject"])
            subject = ie["subject"]
            sid = ie["sid"]
            score = ie["score"]
            relations = ie["relations"]
            for rel in relations:
                fact_candiate = FactCandidate(self.config_options,
                                              query,
                                              subject,
                                              sid,
                                              score,
                                              rel,
                                              relations[rel])
                fact_candiate.get_support_sentence()
                relations[rel]["support"] = list(fact_candiate.support)
        json_path = "/home/hongyul/AMA/support_sentence/" + dataset + "/" + str(query.id)
        codecsDumpJson(json_path, json)
        logger.info("Done extracting wiki data for index %d", idx)

    def extract_wiki_data(self, dataset, idx):
        queries = load_eval_queries(dataset)
        for query in queries:
            if (int(query.id) % 20 != idx):
                continue
            logger.info("Processing query " + str(query.id))
            json = modules.extractor.extract_fact_list_with_entity_linker(dataset, query)
            facts = json["facts"]
            for ie in facts:
                subject = ie["subject"]
                sid = ie["sid"]
                score = ie["score"]
                relations = ie["relations"]
                for rel in relations:
                    fact_candiate = FactCandidate(self.config_options,
                                                  query,
                                                  subject,
                                                  sid,
                                                  score,
                                                  rel,
                                                  relations[rel])
                    fact_candiate.get_support_sentence()
                    relations[rel]["support"] = list(fact_candiate.support)
            json_path = "/home/hongyul/AMA/support_sentence/" + dataset + "/" + str(query.id)
            codecsDumpJson(json_path, json)
        logger.info("Done extracting wiki data for partition %d", idx)

    def extract_fact_candidates(self, dataset):
        queries = load_eval_queries(dataset)
        vocab = set([])
        vocab_trigram = set([])
        sentence_size = 0
        sentence_trigram_size = 0
        candidates = []
        for query in queries:
            logger.info("Processing query " + str(query.id))
            json = modules.extractor.extract_fact_list_with_entity_linker(dataset, query)
            facts = json["facts"]
            query_candidates = []
            for ie in facts:
                subject = ie["subject"]
                sid = ie["sid"]
                score = ie["score"]
                relations = ie["relations"]
                for rel in relations:
                    fact_candiate = FactCandidate(self.config_options,
                                                  query,
                                                  subject,
                                                  sid,
                                                  score,
                                                  rel,
                                                  relations[rel])
                    query_candidates.append(fact_candiate)
                    vocab |= fact_candiate.vocab
                    vocab_trigram |= fact_candiate.vocab_trigram
                    sentence_size = max(fact_candiate.sentence_size, sentence_size)
                    sentence_trigram_size = max(fact_candiate.sentence_trigram_size,
                                                sentence_trigram_size)
            candidates.append(query_candidates)
        d = dict(
            candidates = candidates,
            vocab = vocab,
            vocab_trigram = vocab_trigram,
            sentence_size = sentence_size,
            sentence_trigram_size = sentence_trigram_size,
        )
        return d

    def get_model(self, model_name):
        config_options = self.config_options
        if model_name == "LSTMPointwise":
            model = LSTMPointwise(config_options, model_name)
        elif model_name == "LSTMPointwiseTrigram":
            model = LSTMPointwise(config_options, model_name)
        elif model_name == "LSTMPairwise":
            model = LSTMPairwise(config_options, model_name)
        elif model_name == "LSTMPairwiseTrigram":
            model = LSTMPairwise(config_options, model_name)
        elif model_name == "LSTMJointPairwise":
            model = LSTMJointPairwise(config_options, model_name)
        elif model_name == "LSTMJointPairwiseTrigram":
            model = LSTMJointPairwise(config_options, model_name)
        elif model_name == "DSSMPairwise":
            model = DSSMPairwise(config_options, model_name)
        elif model_name == "EmbeddingJointPairwise":
            model = EmbeddingJointPairwise(config_options, model_name)
        elif model_name == "EmbeddingJointPairwiseTrigram":
            model = EmbeddingJointPairwise(config_options, model_name)
        else:
            logger.warning("Model name " + model_name + " does not exist.")
            model = None
        return model

    def train_model(self, model_name):
        config_options = self.config_options
        train_data = self.extract_fact_candidates("webquestionstrain")
        data = train_data.get('candidates')
        #test_data = self.extract_fact_candidates("webquestionstest")

        if model_name == "LSTMPointwise":
            model = LSTMPointwise(config_options, model_name)
            model.train(data, 28)
        elif model_name == "LSTMPointwiseTrigram":
            model = LSTMPointwise(config_options, model_name)
            model.train(data, 203)
        elif model_name == "LSTMPairwise":
            model = LSTMPairwise(config_options, model_name)
            model.train(data, 28)
        elif model_name == "LSTMPairwiseTrigram":
            model = LSTMPairwise(config_options, model_name)
            model.train(data, 203)
        elif model_name == "LSTMJointPairwise":
            model = LSTMJointPairwise(config_options, model_name)
            model.train(data, 28, 'query_tokens', 'relation_tokens')
        elif model_name == "LSTMJointPairwiseTrigram":
            model = LSTMJointPairwise(config_options, model_name)
            model.train(data, 203, 'query_trigram', 'relation_trigram')
        elif model_name == "DSSMPairwise":
            model = DSSMPairwise(config_options, model_name)
            model.train(data,
                        203,
                        'query_trigram',
                        'relation_trigram',
                        vectorize=vectorize_sentence_one_hot)
        elif model_name == "EmbeddingJointPairwise":
            model = EmbeddingJointPairwise(config_options, model_name)
            model.train(data, 28, 'query_tokens', 'relation_tokens')
        elif model_name == "EmbeddingJointPairwiseTrigram":
            model = EmbeddingJointPairwise(config_options, model_name)
            model.train(data, 203, 'query_trigram', 'relation_trigram')


    def train(self, dataset):
        #lstm_model = LSTMPointwise(self.config_options, 'LSTMPointwise')
        #trigram_model = LSTMPointwise(self.config_options, 'LSTMPointwiseTrigram')
        pairwise_model = self.get_model('LSTMPairwise')
        pairwise_trigram = self.get_model('LSTMPairwiseTrigram')
        jointpairwise = self.get_model('LSTMJointPairwise')
        jointpairwise_trigram = self.get_model('LSTMJointPairwiseTrigram')
        embedding = self.get_model('EmbeddingJointPairwise')
        embedding_trigram = self.get_model('EmbeddingJointPairwiseTrigram')
        #jointpairwise_cnn = self.get_model('CNNPairwise')

        logger.info("Done loading models.")

        queries = load_eval_queries(dataset)
        codecsWriteFile(self.svmTrainingFeatureVectorsFile, "")
        for query in queries:
            logger.info("Processing query " + str(query.id))
            query.dataset = dataset
            json = modules.extractor.extract_fact_list_with_entity_linker(dataset, query)
            facts = json["facts"]
            if len(facts) == 0:
                continue

            query_candidates = []
            for ie in facts:
                subject = ie["subject"]
                sid = ie["sid"]
                score = ie["score"]
                relations = ie["relations"]
                for rel in relations:
                    fact_candiate = FactCandidate(self.config_options,
                                                  query,
                                                  subject,
                                                  sid,
                                                  score,
                                                  rel,
                                                  relations[rel])
                    fact_candiate.extract_features()
                    query_candidates.append(fact_candiate)

            # add lstm feature for all candidates
            pairwise_predictions = pairwise_model.predict(query_candidates, 28).flatten()
            pairwise_trigram_predictions = pairwise_trigram.predict(query_candidates, 203).flatten()
            jointpairwise_predictions = jointpairwise.predict(
                query_candidates,
                28,
                'query_tokens',
                'relation_tokens'
            ).flatten()
            jointpairwise_trigram_predictions = jointpairwise_trigram.predict(
                query_candidates,
                203,
                'query_trigram',
                'relation_trigram'
            ).flatten()
            embedding_predictions = embedding.predict(
                query_candidates,
                28,
                'query_tokens',
                'relation_tokens'
            ).flatten()
            embedding_trigram_predictions = embedding_trigram.predict(
                query_candidates,
                203,
                'query_trigram',
                'relation_trigram'
            ).flatten()
            """
            question_joint_predictions = jointpairwise.predict(
                query_candidates,
                28,
                'query_tokens',
                'top_sentence_with_question'
            ).flatten()
            question_joint_trigram_predictions = jointpairwise_trigram.predict(
                query_candidates,
                203,
                'query_trigram',
                'top_sentence_with_question_trigram'
            ).flatten()
            question_embedding_predictions = embedding.predict(
                query_candidates,
                28,
                'query_tokens',
                'top_sentence_with_question'
            ).flatten()
            question_embedding_trigram_predictions = embedding_trigram.predict(
                query_candidates,
                203,
                'query_tokens',
                'top_sentence_with_question_trigram'
            ).flatten()
            """

            for idx in xrange(len(query_candidates)):
                candidate = query_candidates[idx]
                candidate.add_feature(jointpairwise_predictions[idx])
                candidate.add_feature(jointpairwise_trigram_predictions[idx])
                candidate.add_feature(embedding_predictions[idx])
                candidate.add_feature(embedding_trigram_predictions[idx])
                candidate.add_feature(pairwise_predictions[idx])
                candidate.add_feature(pairwise_trigram_predictions[idx])
                #candidate.add_feature(candidate.support_sentence_score(jointpairwise))
                candidate.add_feature(candidate.top_sentence_score(embedding))
                #candidate.add_feature(question_joint_predictions[idx])
                #candidate.add_feature(question_joint_trigram_predictions[idx])
                #candidate.add_feature(question_embedding_predictions[idx])
                #candidate.add_feature(question_embedding_trigram_predictions[idx])

            self.nomalize_features(query_candidates)
            for candidate in query_candidates:
                codecsWriteFile(self.svmTrainingFeatureVectorsFile,
                                str(candidate.feature_vector),
                                "a")
        self.svm_learn()
        logger.info("Done training svm.")

    def choose_best_candidate(self, candidates, answers):
        count = 0
        best = None
        for candidate in candidates:
            predictions = set(candidate.objects)
            merge = predictions & answers
            if len(merge) > count:
                count = len(merge)
                best = candidate
        return best

    def has_correct_answer(self, candidates, indices, best):
        if best is None:
            return False
        for idx in indices:
            candidate = candidates[idx]
            if candidate.relation == best.relation:
                return True
        return False


    def test(self, dataset):
        pairwise_model = LSTMPairwise(self.config_options, 'LSTMPairwise')
        pairwise_trigram = LSTMPairwise(self.config_options, 'LSTMPairwiseTrigram')
        jointpairwise = self.get_model('LSTMJointPairwise')
        jointpairwise_trigram = self.get_model('LSTMJointPairwiseTrigram')
        embedding = self.get_model('EmbeddingJointPairwise')
        embedding_trigram = self.get_model('EmbeddingJointPairwiseTrigram')
        logger.info("Finish loading models.")

        test_result = self.config_options.get('Test', 'test-result')
        codecsWriteFile(test_result, "")

        support_file = "/home/hongyul/AMA/support_sentence_stats_" + dataset + ".txt"
        not_found_file = "/home/hongyul/AMA/test_result/result_not_found.txt"
        same_file = "/home/hongyul/AMA/test_result/result_diff.txt"
        cover_file = "/home/hongyul/AMA/test_result/result_10.txt"
        above_file = "/home/hongyul/AMA/test_result/result_5-10.txt"
        below_file = "/home/hongyul/AMA/test_result/result_0-5.txt"
        zero_file = "/home/hongyul/AMA/test_result/result_0.txt"
        entity_diff_file = "/home/hongyul/AMA/test_result/result_entity_diff.txt"
        relation_diff_file = "/home/hongyul/AMA/test_result/result_relation_diff.txt"
        support_count_file = "/home/hongyul/AMA/test_result/support_count_file.txt"
        codecsWriteFile(support_file, "")
        codecsWriteFile(not_found_file, "")
        codecsWriteFile(same_file, "")
        codecsWriteFile(cover_file, "")
        codecsWriteFile(above_file, "")
        codecsWriteFile(below_file, "")
        codecsWriteFile(zero_file, "")
        codecsWriteFile(entity_diff_file, "")
        codecsWriteFile(relation_diff_file, "")
        codecsWriteFile(support_count_file, "")

        cover = 0
        num_top2 = 0
        num_top5 = 0
        num_top10 = 0
        not_found = 0
        entity_diff = 0
        relation_diff = 0
        total_support = 0
        queries = load_eval_queries(dataset)
        for query in queries:
            try:
                codecsWriteFile(self.svmTestingFeatureVectorsFile, "")
                query.dataset = dataset
                candidates = []
                support_count = 0

                json = modules.extractor.extract_fact_list_with_entity_linker(dataset, query)
                facts = json["facts"]
                if facts == []:
                    result_line = "\t".join([query.utterance,
                                         str(query.target_result),
                                         str([])]) + "\n"
                    codecsWriteFile(test_result, result_line, "a")
                    continue

                for ie in facts:
                    subject = ie["subject"]
                    sid = ie["sid"]
                    score = ie["score"]
                    relations = ie["relations"]
                    for rel in relations:
                        fact_candidate = FactCandidate(self.config_options,
                                                      query,
                                                      subject,
                                                      sid,
                                                      score,
                                                      rel,
                                                      relations[rel])
                        fact_candidate.extract_features()
                        candidates.append(fact_candidate)
                        support_count += len(fact_candidate.support)
                        total_support += len(fact_candidate.support)
                        codecsWriteFile(support_count_file, str(len(fact_candidate.support)) + "\n", 'a')

                # add model features for all candidates
                # lstm_predictions = lstm_model.predict(candidates, 28).flatten()
                # trigram_predictions = trigram_model.predict(candidates, 203).flatten()

                """
                pairwise_predictions = pairwise_model.predict(candidates, 28).flatten()
                pairwise_trigram_predictions = pairwise_trigram.predict(candidates, 203).flatten()
                jointpairwise_predictions = jointpairwise.predict(
                    candidates,
                    28,
                    'query_tokens',
                    'relation_tokens'
                ).flatten()
                jointpairwise_trigram_predictions = jointpairwise_trigram.predict(
                    candidates,
                    203,
                    'query_trigram',
                    'relation_trigram'
                ).flatten()
                embedding_predictions = embedding.predict(
                    candidates,
                    28,
                    'query_tokens',
                    'relation_tokens'
                ).flatten()
                embedding_trigram_predictions = embedding_trigram.predict(
                    candidates,
                    203,
                    'query_trigram',
                    'relation_trigram'
                ).flatten()
                """

                """
                question_joint_predictions = jointpairwise.predict(
                    candidates,
                    28,
                    'query_tokens',
                    'top_sentence_with_question'
                ).flatten()
                question_joint_trigram_predictions = jointpairwise_trigram.predict(
                    candidates,
                    203,
                    'query_trigram',
                    'top_sentence_with_question_trigram'
                ).flatten()
                question_embedding_predictions = embedding.predict(
                    candidates,
                    28,
                    'query_tokens',
                    'top_sentence_with_question'
                ).flatten()
                question_embedding_trigram_predictions = embedding_trigram.predict(
                    candidates,
                    203,
                    'query_tokens',
                    'top_sentence_with_question_trigram'
                ).flatten()
                """

                for idx in xrange(len(candidates)):
                    candidate = candidates[idx]
                    candidate.add_feature(0)
                    candidate.add_feature(0)
                    candidate.add_feature(0)
                    candidate.add_feature(0)
                    candidate.add_feature(0)
                    candidate.add_feature(0)
                    candidate.add_feature(0)

                    """
                    candidate.add_feature(jointpairwise_predictions[idx])
                    candidate.add_feature(jointpairwise_trigram_predictions[idx])
                    candidate.add_feature(embedding_predictions[idx])
                    candidate.add_feature(embedding_trigram_predictions[idx])
                    candidate.add_feature(pairwise_predictions[idx])
                    candidate.add_feature(pairwise_trigram_predictions[idx])
                    candidate.add_feature(0)
                    """
                    #candidate.add_feature(candidate.support_sentence_score(jointpairwise))
                    #candidate.add_feature(candidate.top_sentence_score(embedding))
                    #candidate.add_feature(question_joint_predictions[idx])
                    #candidate.add_feature(question_joint_trigram_predictions[idx])
                    #candidate.add_feature(question_embedding_predictions[idx])
                    #candidate.add_feature(question_embedding_trigram_predictions[idx])

                """
                support_stats_file = "/home/hongyul/AMA/support_sentence_stat/" + dataset + "/" + str(query.id)
                codecsWriteFile(support_stats_file, "")
                for candidate in candidates:
                    stats = [query.utterance,
                        str(support_count),
                        candidate.graph_str,
                        str(len(candidate.support)),
                        candidate.top_sentence]
                    codecsWriteFile(support_stats_file, "\t".join(stats) + "\n", 'a')
                """

                self.nomalize_features(candidates)
                for candidate in candidates:
                    codecsWriteFile(self.svmTestingFeatureVectorsFile,
                                    str(candidate.feature_vector),
                                    "a")
                self.svm_rank()

                # Choose answers from candidates
                answers = set(query.target_result)
                scores = [float(n) for n in codecsReadFile(self.svmFactCandidateScores).strip().split("\n")]
                idx = np.argmax(scores)
                top2 = np.argsort(scores)[::-1][:2]
                top5 = np.argsort(scores)[::-1][:5]
                top10 = np.argsort(scores)[::-1][:10]

                best_candidate = candidates[idx]
                best = self.choose_best_candidate(candidates, answers)

                if best is None:
                    best_relation = "EMPTY"
                    best_subject = "NONE"
                    stats = [str(query.id),
                             query.utterance,
                             str(support_count),
                             "EMPTY",
                             "0",
                             best_candidate.graph_str,
                             str(len(best_candidate.support))]
                else:
                    best_relation = best.relation
                    best_subject = best.subject
                    stats = [str(query.id),
                             query.utterance,
                             str(support_count),
                             best.graph_str,
                             str(len(best.support)),
                             best_candidate.graph_str,
                             str(len(best_candidate.support))]
                codecsWriteFile(support_file, "\t".join(stats) + "\n", "a")

                if best_candidate.relation == best_relation:
                    cover += 1
                if self.has_correct_answer(candidates, top2, best):
                    num_top2 += 1
                if self.has_correct_answer(candidates, top5, best):
                    num_top5 += 1
                if self.has_correct_answer(candidates, top10, best):
                    num_top10 += 1

                best_predictions = list(set(best_candidate.objects))
                if len(best_predictions) > 5:
                    limit_predictions = best_predictions[:5] + ["..."]
                else:
                    limit_predictions = best_predictions

                result_line = "\t".join([str(query.id) + query.utterance,
                                         str(query.target_result),
                                         str(list(best_predictions)),
                                         str(best_candidate.f1),
                                         best_relation,
                                         best_candidate.relation,
                                         best_subject,
                                         best_candidate.subject]) + "\n"
                codecsWriteFile(test_result, result_line, "a")
                #print("Processing query ", str(query.id), cover, " " * 10 + "\r", end="")
                message = " ".join(["Processing query",
                                    str(query.id) + ":",
                                    str(cover),
                                    str(num_top2),
                                    str(num_top5),
                                    str(num_top10)])
                logger.info(message)

                content = "\t".join([str(query.id) + query.utterance,
                                         str(query.target_result),
                                         str(list(limit_predictions)),
                                         str(best_candidate.f1),
                                         best_relation,
                                         best_candidate.relation,
                                         best_subject,
                                         best_candidate.subject]) + "\n"

                """
                if best is None:
                    content += "Empty\n"
                else:
                    content += str(best)
                content += "Top5\n"
                for idx in top5:
                    candidate = candidates[idx]
                    content += str(candidate)
                content += "\n"
                if best_candidate.f1 == 1.0:
                    codecsWriteFile(cover_file, content, "a")
                elif best_candidate.f1 >= 0.5:
                    codecsWriteFile(above_file, content, "a")
                elif best_candidate.f1 > 0:
                    codecsWriteFile(below_file, content, "a")
                else:
                    codecsWriteFile(zero_file, content, "a")
                if best is None:
                    codecsWriteFile(not_found_file, content, "a")
                    not_found += 1
                elif best_subject != best_candidate.subject:
                    codecsWriteFile(entity_diff_file, content, "a")
                    entity_diff += 1
                elif best_candidate.relation != best_relation:
                    codecsWriteFile(same_file, content, "a")
                    relation_diff += 1
                """

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

        print("")
        print(cover)
        print(num_top5)
        print(num_top10)

        logger.info("Best candidate not found: %d", not_found)
        logger.info("Entity diff count: %d", entity_diff)
        logger.info("Relation diff count: %d", relation_diff)
        logger.info("Total support sentence count: %d", total_support)






