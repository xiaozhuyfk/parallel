from __future__ import print_function

import logging
import globals
import modules
from evaluation import load_eval_queries
from util import codecsWriteFile, codecsReadFile, codecsLoadJson, kstem, codecsDumpJson
import codecs
import re
import numpy as np
import random
import json
import os
import tensorflow as tf
from memory_network import MemN2N
from data_utils import vectorize_data, memory_data, selective_data
from itertools import chain
from sklearn import cross_validation, metrics

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


stems = {}

def load_data_from_disk(query, path):
        id = query.id
        file_path = path + str(id)
        if os.path.isfile(file_path):
            d = codecsLoadJson(file_path)
            return d
        else:
            return None

def tokenize_term(t):
    return re.sub('[?!@#$%^&*,()_+=\'\d\./]', '', t).lower()

def get_stem(t):
    global stems
    if t in stems:
        return stems[t]
    else:
        stem = kstem(t)
        stems[t] = stem
        return stem

def process_data(dataset, path):
    queries = load_eval_queries(dataset)
    sentence_size = 0
    vocab = set([])

    for query in queries:
        logger.info("Processing question " + str(query.id))

        d = load_data_from_disk(query, path)
        if d is not None:
            q = d.get("query")
            s = d.get("story")
            a = d.get("answer")
            for ss in s:
                if len(ss) > sentence_size:
                    sentence_size = len(ss)
            if len(q) > sentence_size:
                sentence_size = len(q)
            vocab |= set(list(chain.from_iterable(s)) + q + a)
            continue

    return vocab, sentence_size

    """
        data_path = path + str(query.id)
        codecsWriteFile(data_path, "")

        facts = modules.extractor.extract_fact_list_with_entity_linker(query)
        question = query.utterance.lower()[:-1]
        tokens = [tokenize_term(t) for t in question.split()]
        story = []
        S = []
        R = []
        O = []
        y = []
        sstory = []
        SS = []
        RR = []
        OO = []
        yy = []
        answer = query.target_result

        story_set = set([])
        for fact in facts:
            sid, s, r, oid, o = fact
            if (o.startswith("g.")):
                continue
            relations = re.split("\.\.|\.", r)[-2:]
            rels = [tokenize_term(e) for t in relations for e in re.split('\.\.|\.|_', t)]
            subjects = [re.sub('[?!@#$%^&*,()_+=\'/]', '', t).lower() for t in s.split()]
            objects = [re.sub('[?!@#$%^&*,()_+=\']', '', t).lower() for t in o.split()]
            if (len(objects) > 10):
                continue
            sentence = subjects + rels #+ objects
            story.append(sentence)
            S.append(subjects)
            R.append(rels)
            O.append(objects)
            y.append((o in answer) * 1.0)

            s = " ".join(sentence)
            if s not in story_set:
                if len(sentence) > sentence_size:
                    sentence_size = len(sentence)
                sstory.append(sentence)
                story_set.add(s)

                SS.append(subjects)
                RR.append(rels)
                OO.append(objects)
                yy.append((o in answer) * 1.0)

        d = {"query" : tokens,
             "story" : story,
             "answer" : answer,
             "S": S,
             "R": R,
             "O": O,
             "y": y,
             "sstory" : sstory,
             "SS" : SS,
             "RR" : RR,
             "OO" : OO,
             "yy" : yy}

        if len(tokens) > sentence_size:
            sentence_size = len(tokens)

        with codecs.open(data_path, mode='w', encoding='utf-8') as f:
            json.dump(d, f, indent=4)
    logger.info("Longest sentence size: " + str(sentence_size))
    """


tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 1, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 30, "Number of epochs to trai`n for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

def load_data(dataset):
    """
    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    training_data = config_options.get('Train', 'training-data')
    testing_data = config_options.get('Test', 'testing-data')

    vocab1, size1 = process_data("webquestionstrain", training_data)
    vocab2, size2 = process_data("webquestionstest", testing_data)
    vocab = sorted(vocab1 | vocab2)
    codecsDumpJson(vocab_file, vocab)

    print(size1)
    print(size2)
    """

    queries = load_eval_queries(dataset)
    for query in queries:
        logger.info("Processing question " + str(query.id))
        modules.extractor.extract_fact_list_with_entity_linker(dataset, query)
    logger.info("Done extracting new fact list.")


def training_progress_message(epoch, epochs, query_id, total, loss):
    progress = ("Progress: %.2f" % (float(query_id + 1) / total * 100)) + "%"
    message = "Processing question " + str(query_id) + ". "
    training = "Epoch %d/%d: loss = %f. " % (epoch, epochs, loss)

    return message + training + progress + " "*10 + "\r"

def randomize_input(trainS, trainQ, trainA):
    shuffleS = []
    shuffleQ = []
    shuffleA = []
    indices = range(len(trainS))
    random.shuffle(indices)

    for i in indices:
        shuffleS.append(trainS[i])
        shuffleQ.append(trainQ[i])
        shuffleA.append(trainA[i])

    trainS = np.array(shuffleS)
    trainQ = np.array(shuffleQ)
    trainA = np.array(shuffleA)
    return trainS, trainQ, trainA

def train(dataset):
    load_data(dataset)
    exit()

    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    training_data = config_options.get('Train', 'training-data')
    model_file = config_options.get('Train', 'model-file')

    vocab = codecsLoadJson(vocab_file)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    batch_size = FLAGS.batch_size
    #sentence_size = 28
    sentence_size = 17
    memory_size = 10
    vocab_size = len(word_idx) + 1

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                       epsilon=FLAGS.epsilon)
    with tf.Session() as sess:
        model = MemN2N(batch_size,
                       vocab_size,
                       sentence_size,
                       memory_size,
                       64,
                       session=sess,
                       hops=FLAGS.hops,
                       max_grad_norm=FLAGS.max_grad_norm,
                       optimizer=optimizer)

        queries = load_eval_queries(dataset)
        data = []
        for i in xrange(len(queries)):
            query = queries[i]
            d = load_data_from_disk(query, training_data)
            data.append(d)
        S, Q, A = selective_data(data, word_idx, sentence_size, memory_size)
        trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=FLAGS.random_state)
        n_train = trainS.shape[0]
        n_val = valS.shape[0]

        print("Training Size", n_train)
        print("Validation Size", n_val)

        train_labels = np.argmax(trainA, axis=1)
        val_labels = np.argmax(valA, axis=1)

        tf.set_random_seed(FLAGS.random_state)
        batch_size = FLAGS.batch_size
        batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
        batches = [(start, end) for start, end in batches]

        for t in range(1, FLAGS.epochs+1):
            np.random.shuffle(batches)
            total_cost = 0.0
            for start, end in batches:
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                cost_t = model.batch_fit(s, q, a)
                total_cost += cost_t

            if t % FLAGS.evaluation_interval == 0:
                train_preds = []
                for start in range(0, n_train, batch_size):
                    end = start + batch_size
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    pred = model.predict(s, q)
                    train_preds += list(pred)

                val_preds = model.predict(valS, valQ)
                train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
                val_acc = metrics.accuracy_score(val_preds, val_labels)

                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                print('-----------------------')
        model.save_model(model_file)

        """
        for t in range(1, FLAGS.epochs+1):
            loss = 0
            for i in xrange(len(queries)):
                #if i >= 2000:
                #    break
                query = queries[i]
                #print("Processing question " + str(query.id) + "            \r", end="")
                print(training_progress_message(t, FLAGS.epochs, i+1, len(queries), loss), end="")
                d = load_data_from_disk(query, training_data)
                data.append(d)

                if len(data) >= 100:
                    trainS, trainQ, trainA = selective_data(data, word_idx, sentence_size, memory_size)
                    trainS, trainQ, trainA = randomize_input(trainS, trainQ, trainA)
                    loss = model.batch_fit(trainS, trainQ, trainA)
                    #message = "Epoch %d/%d: loss = %f" % (t, FLAGS.epochs, loss)
                    #logger.info(message)
                    data = []

            if len(data) > 0:
                trainS, trainQ, trainA = selective_data(data, word_idx, sentence_size, memory_size)
                trainS, trainQ, trainA = randomize_input(trainS, trainQ, trainA)
                loss = model.batch_fit(trainS, trainQ, trainA)
            print("")
            message = "Epoch %d/%d: loss = %f" % (t, FLAGS.epochs, loss)
            logger.info(message)
        model.save_model(model_file)
        """

def process_facts(facts):
    w2v = modules.w2v
    F = []
    for fact in facts:
        sid, s, r, oid, o = fact
        relations = re.split("\.\.|\.", r)[:-2]
        rels = w2v.transform_seq([e for t in relations for e in re.split('\.\.|\.|_', t)])
        subjects = w2v.transform_seq([re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in s.split()])
        objects = w2v.transform_seq([re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in o.split()])
        F.append((subjects, rels, objects))
    return F


def test_iter(model, S, Q, indices):
    n, memory_size, sentence_size = S.shape
    predictions = model.predict(S, Q)

    if n == 1:
        idx = predictions[0]
        return indices[idx]

    new_index = []
    ss = []
    for i in xrange(n):
        best = predictions[i]
        ss.append(S[i][best])
        best = i * memory_size + best
        new_index.append(indices[best])

    testS = []
    testQ = []
    nn = n / memory_size + 1
    for i in xrange(nn):
        stmp = ss[i * memory_size : (i+1) * memory_size]
        lm = max(0, memory_size - len(stmp))
        for _ in xrange(lm):
            stmp.append([0] * sentence_size)
        testS.append(stmp)
        testQ.append(Q[0])
    testS = np.array(testS)
    testQ = np.array(testQ)

    lm = max(0, memory_size * nn - len(new_index))
    new_index += [-1] * lm

    return test_iter(model, testS, testQ, new_index)





def test(dataset):
    load_data(dataset)
    exit()

    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    testing_data = config_options.get('Test', 'testing-data')
    model_file = config_options.get('Train', 'model-file')
    test_result = config_options.get('Test', 'test-result')

    vocab = codecsLoadJson(vocab_file)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    batch_size = FLAGS.batch_size
    sentence_size = 17
    memory_size = 10
    vocab_size = len(word_idx) + 1

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                       epsilon=FLAGS.epsilon)

    with tf.Session() as sess:
        model = MemN2N(batch_size,
                       vocab_size,
                       sentence_size,
                       memory_size,
                       64,
                       session=sess,
                       hops=FLAGS.hops,
                       max_grad_norm=FLAGS.max_grad_norm,
                       optimizer=optimizer)
        model.load_model(model_file)

        queries = load_eval_queries(dataset)
        data = []
        for i in xrange(len(queries)):
            query = queries[i]
            d = load_data_from_disk(query, testing_data)
            data.append(d)
        S, Q, A = selective_data(data, word_idx, sentence_size, memory_size)

        test_labels = np.argmax(A, axis=1)
        test_preds = model.predict(S, Q)
        test_acc = metrics.accuracy_score(test_preds, test_labels)
        print("Testing Accuracy:", test_acc)

        aqqu_labels = np.zeros(len(test_labels))
        aqqu_acc = metrics.accuracy_score(aqqu_labels, test_labels)
        print("Aqqu Accuracy:", aqqu_acc)

        """
        data = []
        correct = 0
        for i in xrange(len(queries)):
            query = queries[i]
            #logger.info("Processing question " + str(query.id))
            d = load_data_from_disk(query, testing_data)
            data.append(d)
            if len(data) >= 100:
                trainS, trainQ, trainA = selective_data(data, word_idx, sentence_size, memory_size)
                #trainS, trainQ, trainA = randomize_input(trainS, trainQ, trainA)
                logger.info("Done extracting vectors.")
                predictions = model.predict_proba(trainS, trainQ)


                for j in xrange(len(predictions)):
                    arr = predictions[j]
                    idx = np.argmax(arr)
                    print(arr[idx])
                    indicator = trainA[j][idx]
                    if indicator == 1.0:
                        correct += 1
                logger.info("Current accuracy = " + str(float(correct) / i))
                data = []
        logger.info("Accuracy = " + str(float(correct) / len(queries)))
        """

        """
        queries = load_eval_queries(dataset)
        codecsWriteFile(test_result, "")
        for i in xrange(len(queries)):
            query = queries[i]
            question = query.utterance.lower()
            answer = query.target_result

            logger.info("Processing question " + str(query.id))
            d = load_data_from_disk(query, testing_data)
            trainS, trainQ, trainA = memory_data([d], word_idx, sentence_size, memory_size)
            logger.info("Done extracting vectors.")
            #predictions = model.predict_proba(trainS, trainQ).flatten()
            #idx = np.argmax(predictions)
            idx = test_iter(model, trainS, trainQ, range(len(trainS) * memory_size))

            if idx >= len(d.get("O")) or idx == -1:
                result = []
                result_line = "\t".join([str(query.id) + question, str(answer), str(result)]) + "\n"
            else:
                best_s = " ".join(d.get("S")[idx])
                best_r = " ".join(d.get("R")[idx])
                result = []
                for i in xrange(len(d.get("O"))):
                    s = " ".join(d.get("S")[i])
                    r = " ".join(d.get("R")[i])

                    if s == best_s and r == best_r:
                        result.append(" ".join(d.get("O")[i]))
                result_line = "\t".join([str(query.id) + question, str(answer), str(result)]) + "\n"
            #print(result_line)
            codecsWriteFile(test_result, result_line, "a")
        """

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Choose to learn or test AMA')

    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use')
    subparsers = parser.add_subparsers(help='command help')
    train_parser = subparsers.add_parser('train', help='Train memory network')
    train_parser.add_argument('dataset',
                              help='The dataset to train.')
    train_parser.set_defaults(which='train')

    test_parser = subparsers.add_parser('test', help='Test memory network')
    test_parser.add_argument('dataset',
                             help='The dataset to test')
    test_parser.set_defaults(which='test')

    process_parser = subparsers.add_parser('process', help="Process training data")
    process_parser.add_argument('dataset',
                                help='Training data file')
    process_parser.set_defaults(which='process')

    args = parser.parse_args()

    # Read global config
    globals.read_configuration(args.config)

    # Load modules
    modules.init_from_config(args)

    if args.which == 'train':
        train(args.dataset)
    elif args.which == 'test':
        test(args.dataset)
    elif args.which == 'process':
        process_trainingdata(args.dataset)


if __name__ == '__main__':
    main()