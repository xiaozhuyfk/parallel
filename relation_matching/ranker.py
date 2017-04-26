import logging
import modules
import time
from util import (
    codecsWriteFile,
    codecsReadFile,
)
import datetime
import subprocess
import re
import numpy as np
from model import (
    LSTMPointwise,
    LSTMPairwise,
    LSTMJointPairwise,
    DSSMPairwise,
    EmbeddingJointPairwise,
)

logger = logging.getLogger(__name__)


def tokenize_term(t):
    return re.sub('[?!@#$%^&*,()_+=\'\d\./;]', '', t).lower()


class FeatureVector(object):

    def __init__(self, relevance, candidate):
        self.query_id = 0
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
    def __init__(self, config_options, question, subject, sid, score, relation, response):
        self.config_options = config_options
        self.question = question

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
        self.relation_tokens = [tokenize_term(e)
                                for t in relations
                                for e in re.split("\.\.|\.|_", t)]

        # character tri-gram
        self.query_trigram = ngramize(self.query_tokens, 3)
        self.subject_trigram = ngramize(self.subject_tokens, 3)
        self.relation_trigram = ngramize(self.relation_tokens, 3)

        self.relevance = 0

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

        self.f1 = 0

        graph_tokens = [self.subject,
                        self.relation,
                        str(self.objects[:5]).encode("utf-8")]
        self.graph_str = " --> ".join(graph_tokens)


    def __str__(self):
        graph_tokens = [" ".join(self.subject_tokens),
                        " ".join(self.relation_tokens),
                        str(self.objects[:5])]
        graph_str = " --> ".join(graph_tokens)
        self.message = "Entity Score = %f, F1 = %f, graph = %s\n" % (self.score, self.f1, graph_str)

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
        vector = FeatureVector(0, self)
        self.feature_vector = vector

        # Add entity linking score
        self.add_feature(float(self.score))

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

        self.pairwise_model = self.get_model('LSTMPairwise')
        self.pairwise_trigram = self.get_model('LSTMPairwiseTrigram')
        self.jointpairwise = self.get_model('LSTMJointPairwise')
        self.jointpairwise_trigram = self.get_model('LSTMJointPairwiseTrigram')
        self.embedding = self.get_model('EmbeddingJointPairwise')
        self.embedding_trigram = self.get_model('EmbeddingJointPairwiseTrigram')
        logger.info("Done loading models.")

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

    def svm_rank(self, testing_path, scores_path):
        logger.info("Start SVM Ranking ...")
        cmd = [self.svmRankClassifyPath,
               testing_path,
               self.svmRankModelFile,
               scores_path]
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

    def rank(self, question):
        question = question.lower()
        timestamp = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        filename = question.encode('utf-8')[:10] + ' ' + timestamp
        testing_path = '/home/ubuntu/parallel/svm_result/' + filename + '.LeToRTest'
        scores_path = '/home/ubuntu/parallel/svm_result/' + filename + '.RankScore'
        codecsWriteFile(testing_path, "")

        json = modules.facts_extractor.extract_fact_list_with_entity_linker(question)
        if json == []:
            return []

        start_time = time.time()
        candidates = []
        for ie in json:
            subject = ie["subject"]
            sid = ie["sid"]
            score = ie["score"]
            relations = ie["relations"]
            for rel in relations:
                fact_candidate = FactCandidate(self.config_options,
                                              question,
                                              subject,
                                              sid,
                                              score,
                                              rel,
                                              relations[rel])
                fact_candidate.extract_features()
                candidates.append(fact_candidate)
        duration = (time.time() - start_time) * 1000
        logger.info("Feature Extraction time: %.2f ms." % duration)

        start_time = time.time()
        pairwise_predictions = self.pairwise_model.predict(candidates, 28).flatten()
        pairwise_trigram_predictions = self.pairwise_trigram.predict(candidates, 203).flatten()
        jointpairwise_predictions = self.jointpairwise.predict(
            candidates,
            28,
            'query_tokens',
            'relation_tokens'
        ).flatten()
        jointpairwise_trigram_predictions = self.jointpairwise_trigram.predict(
            candidates,
            203,
            'query_trigram',
            'relation_trigram'
        ).flatten()
        embedding_predictions = self.embedding.predict(
            candidates,
            28,
            'query_tokens',
            'relation_tokens'
        ).flatten()
        embedding_trigram_predictions = self.embedding_trigram.predict(
            candidates,
            203,
            'query_trigram',
            'relation_trigram'
        ).flatten()
        duration = (time.time() - start_time) * 1000
        logger.info("Relation Score Computation time: %.2f ms." % duration)


        start_time = time.time()
        for idx in xrange(len(candidates)):
            candidate = candidates[idx]
            candidate.add_feature(jointpairwise_predictions[idx])
            candidate.add_feature(jointpairwise_trigram_predictions[idx])
            candidate.add_feature(embedding_predictions[idx])
            candidate.add_feature(embedding_trigram_predictions[idx])
            candidate.add_feature(pairwise_predictions[idx])
            candidate.add_feature(pairwise_trigram_predictions[idx])

        self.nomalize_features(candidates)
        for candidate in candidates:
            codecsWriteFile(testing_path,
                            str(candidate.feature_vector),
                            "a")
        self.svm_rank(testing_path, scores_path)
        duration = (time.time() - start_time) * 1000
        logger.info("SVM Ranking time: %.2f ms." % duration)

        # Choose answers from candidates
        scores = [float(n) for n in codecsReadFile(scores_path).strip().split("\n")]
        top5 = np.argsort(scores)[::-1][:5]
        return [candidates[idx] for idx in top5]


    def extract_nodes_and_links(self, candidates):
        nodes = []
        links = []
        subjects = set([])
        for candidate in candidates:
            if (candidate.sid not in subjects):
                subjects.add(candidate.sid)

                subject_node = dict(
                    match = 1.0,
                    name = candidate.subject,
                    artist = candidate.sid,
                    id = candidate.sid,
                    playcount = 10,
                )
                nodes.append(subject_node)

            relation_node = dict(
                match = 1.0,
                name = "-".join(candidate.relation_tokens),
                artist = candidate.relation,
                id = candidate.relation,
                playcount = 8,
            )
            if candidate.relation not in subjects:
                subjects.add(candidate.relation)
                nodes.append(relation_node)

            subject_relation = dict(
                source = candidate.sid,
                target = candidate.relation,
            )
            links.append(subject_relation)

            for idx in xrange(len(candidate.objects)):
                object_id = candidate.oid[idx] + "-" + candidate.objects[idx] + "-" + candidate.relation
                object_node = dict(
                    match = 1.0,
                    name = candidate.objects[idx],
                    artist = candidate.oid[idx],
                    id = object_id,
                    playcount = 5
                )
                if object_id not in subjects:
                    subjects.add(object_id)
                    nodes.append(object_node)

                relation = dict(
                    source = candidate.relation,
                    target = object_id,
                )
                links.append(relation)

        result = dict(
            nodes = nodes,
            links = links,
        )

        return result

    def network(self, question):
        question = question.lower()
        timestamp = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        filename = question.encode('utf-8')[:10] + ' ' + timestamp
        testing_path = '/home/ubuntu/parallel/svm_result/' + filename + '.LeToRTest'
        scores_path = '/home/ubuntu/parallel/svm_result/' + filename + '.RankScore'
        codecsWriteFile(testing_path, "")

        json = modules.facts_extractor.extract_fact_list_with_entity_linker(question)
        if json == []:
            return []

        start_time = time.time()
        candidates = []
        for ie in json:
            subject = ie["subject"]
            sid = ie["sid"]
            score = ie["score"]
            relations = ie["relations"]
            for rel in relations:
                fact_candidate = FactCandidate(self.config_options,
                                              question,
                                              subject,
                                              sid,
                                              score,
                                              rel,
                                              relations[rel])
                fact_candidate.extract_features()
                candidates.append(fact_candidate)
        duration = (time.time() - start_time) * 1000
        logger.info("Feature Extraction time: %.2f ms." % duration)

        start_time = time.time()
        pairwise_predictions = self.pairwise_model.predict(candidates, 28).flatten()
        pairwise_trigram_predictions = self.pairwise_trigram.predict(candidates, 203).flatten()
        jointpairwise_predictions = self.jointpairwise.predict(
            candidates,
            28,
            'query_tokens',
            'relation_tokens'
        ).flatten()
        jointpairwise_trigram_predictions = self.jointpairwise_trigram.predict(
            candidates,
            203,
            'query_trigram',
            'relation_trigram'
        ).flatten()
        embedding_predictions = self.embedding.predict(
            candidates,
            28,
            'query_tokens',
            'relation_tokens'
        ).flatten()
        embedding_trigram_predictions = self.embedding_trigram.predict(
            candidates,
            203,
            'query_trigram',
            'relation_trigram'
        ).flatten()
        duration = (time.time() - start_time) * 1000
        logger.info("Relation Score Computation time: %.2f ms." % duration)


        start_time = time.time()
        for idx in xrange(len(candidates)):
            candidate = candidates[idx]
            candidate.add_feature(jointpairwise_predictions[idx])
            candidate.add_feature(jointpairwise_trigram_predictions[idx])
            candidate.add_feature(embedding_predictions[idx])
            candidate.add_feature(embedding_trigram_predictions[idx])
            candidate.add_feature(pairwise_predictions[idx])
            candidate.add_feature(pairwise_trigram_predictions[idx])

        self.nomalize_features(candidates)
        for candidate in candidates:
            codecsWriteFile(testing_path,
                            str(candidate.feature_vector),
                            "a")
        self.svm_rank(testing_path, scores_path)
        duration = (time.time() - start_time) * 1000
        logger.info("SVM Ranking time: %.2f ms." % duration)

        # Choose answers from candidates
        scores = [float(n) for n in codecsReadFile(scores_path).strip().split("\n")]
        top5 = np.argsort(scores)[::-1][:5]

        result = self.extract_nodes_and_links(candidates)

        return result
