from __future__ import print_function

import logging
import globals
import modules
from evaluation import load_eval_queries
from util import codecsWriteFile, codecsReadFile, codecsLoadJson
import codecs
import re
from keras.layers import Input, LSTM, Dense, Embedding, Merge
from keras.models import Model, model_from_json, Sequential
import numpy as np
import random
from alphabet import Alphabet
from question_embedding import QuestionEncoder
import json
from itertools import chain
import os
import tensorflow as tf
from memory_network import MemN2N
from data_utils import vectorize_data, memory_data, selective_data

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def save_model_to_file(model, struct_file, weights_file):
    # save model structure
    model_struct = model.to_json()
    open(struct_file, 'w').write(model_struct)

    # save model weights
    model.save_weights(weights_file, overwrite=True)


def load_model(struct_file, weights_file):
    model = model_from_json(open(struct_file, 'r').read())
    model.compile(loss="binary_crossentropy", optimizer='rmsprop')
    model.load_weights(weights_file)

    return model


def transform_to_vectors(tokens, input_dim):
    vectors = np.zeros((input_dim, 300))
    valid = []
    for word in tokens:
        v = modules.w2v.transform(word)
        if v is not None:
            valid.append(v)

    for i in xrange(len(valid)):
        idx = input_dim - len(valid) + i
        vectors[idx] = valid[i]

    return vectors


def process_line(line, input_dim):
    words = line.strip().split()
    label = float(words[-1])
    vectors = transform_to_vectors(words[:-1], input_dim)
    return vectors, label

def generate_data_from_file(path, input_dim):
    f = codecs.open(path, mode="rt", encoding="utf-8")
    while True:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            x, y = process_line(line, input_dim)
            yield (np.array([x]), np.array([y]))
    f.close()



def process_trainingdata(dataset):
    config_options = globals.config
    training_data = config_options.get('Train', 'training-data')

    queries = load_eval_queries(dataset)
    codecsWriteFile(training_data, "")
    count = 0
    length = 0
    for query in queries:
        facts = modules.extractor.extract_fact_list_with_entity_linker(query)

        question = query.utterance.lower()[:-1]
        logger.info("Processing question " + str(query.id))
        hasAnwer = False

        answer = query.target_result
        correct = []
        wrong = []
        for fact in facts:
            sid, s, r, oid, o = fact
            if (o.startswith("g.")):
                continue
            if (o in answer):
                #line = question + "\t" + "\t".join(fact) + "\t" + "1" + "\n"
                #codecsWriteFile("trainingdata", line, 'a')
                correct.append(fact)
                hasAnwer = True
            else:
                #line = question + "\t" + "\t".join(fact) + "\t" + "0" + "\n"
                #codecsWriteFile("trainingdata", line, 'a')
                wrong.append(fact)

        if not hasAnwer:
            logger.info(question + " does not have an answer.")
            count += 1

        for fact in correct:
            sid, s, r, oid, o = fact
            relations = re.split("\.\.|\.", r)[-2:]
            rels = [e for t in relations for e in re.split('\.\.|\.|_', t)]

            tokens = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in question.split()]
            subjects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in s.split()]
            objects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in o.split()]
            if (len(objects) > 10):
                continue

            line = "\t".join(tokens + subjects + rels + objects + ["1.0"]) + "\n"
            if (len(tokens + subjects + rels + objects) > length):
                length = len(tokens + subjects + rels + objects)
            codecsWriteFile(training_data, line, "a")

        sample = wrong
        if len(correct) == 0:
            if len(wrong) > 20:
                sample = random.sample(wrong, 20)
        elif len(correct) * 20 > len(wrong):
            sample = wrong
        else:
            sample = random.sample(wrong, len(correct) * 20)

        for fact in sample:
            sid, s, r, oid, o = fact
            relations = re.split("\.\.|\.", r)[:-2]
            rels = [e for t in relations for e in re.split('\.\.|\.|_', t)]

            tokens = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in question.split()]
            subjects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in s.split()]
            objects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in o.split()]
            if (len(objects) > 10):
                continue

            line = "\t".join(tokens + subjects + rels + objects + ["0.0"]) + "\n"
            if (len(tokens + subjects + rels + objects) > length):
                length = len(tokens + subjects + rels + objects)
            codecsWriteFile(training_data, line, "a")

    logger.info(str(count) + " questions do not have answers.")
    logger.info("Longest vector of length " + str(length))


def load_data_from_disk(query, path):
        id = query.id
        file_path = path + str(id)
        if os.path.isfile(file_path):
            d = codecsLoadJson(file_path)
            #with codecs.open(file_path, "wt", encoding='utf-8') as f:
            #    d = json.load(f)
            return d
        else:
            return None

def process_data(dataset, path):
    queries = load_eval_queries(dataset)
    for query in queries:
        logger.info("Processing question " + str(query.id))
        """
        d = load_data_from_disk(query, path)
        if d is not None:
            q = d.get("query")
            s = d.get("story")
            a = d.get("answer")
            vocab |= set(list(chain.from_iterable(s)) + q + a)
        """

        data_path = path + str(query.id)
        codecsWriteFile(data_path, "")

        facts = modules.extractor.extract_fact_list_with_entity_linker(query)
        question = query.utterance.lower()[:-1]
        tokens = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in question.split()]
        story = []
        S = []
        R = []
        O = []
        y = []
        answer = query.target_result

        for fact in facts:
            sid, s, r, oid, o = fact
            if (o.startswith("g.")):
                continue
            relations = re.split("\.\.|\.", r)[-2:]
            rels = [e for t in relations for e in re.split('\.\.|\.|_', t)]
            subjects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in s.split()]
            objects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in o.split()]
            if (len(objects) > 10):
                continue
            sentence = subjects + rels + objects
            story.append(sentence)
            S.append(subjects)
            R.append(rels)
            O.append(objects)
            y.append((o in answer) * 1.0)

        d = {"query" : tokens,
             "story" : story,
             "answer" : answer,
             "S": S,
             "R": R,
             "O": O,
             "y": y}

        with codecs.open(data_path, mode='w', encoding='utf-8') as f:
            json.dump(d, f, indent=4)


tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 10, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

def load_data():
    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    training_data = config_options.get('Train', 'training-data')
    testing_data = config_options.get('Test', 'testing-data')
    model_file = config_options.get('Train', 'model-file')

    #process_data("webquestionstrain", training_data)
    #process_data("webquestionstest", testing_data)

    vocab = codecsLoadJson(vocab_file)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    batch_size = FLAGS.batch_size
    sentence_size = 28
    memory_size = 20
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

        train_queries = load_eval_queries("webquestionstrain")
        data = []
        for i in xrange(len(train_queries)):
            query = train_queries[i]
            logger.info("Processing question " + str(query.id))
            d = load_data_from_disk(query, training_data)
            data.append(d)

            if len(data) >= 100:
                trainS, trainQ, trainA = memory_data(data, word_idx, sentence_size, memory_size)
                logger.info("Done loading memory vectors.")
                for t in range(1, FLAGS.epochs+1):
                    model.batch_fit(trainS, trainQ, trainA)
                data = []
        if len(data) > 0:
            trainS, trainQ, trainA = memory_data(data, word_idx, sentence_size, memory_size)
            logger.info("Done loading memory vectors.")
            for t in range(1, FLAGS.epochs+1):
                model.batch_fit(trainS, trainQ, trainA)
        model.save_model(model_file)

    #vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    #word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    """
    sentence_size_train, memory_size_train = process_data("webquestionstrain", training_data)
    sentence_size_test, memory_size_test = process_data("webquestionstest", testing_data)

    logger.info("Sentence size for training data: " + str(sentence_size_train))
    logger.info("Memory size for training data: " + str(memory_size_train))
    logger.info("Sentence size for test data: " + str(sentence_size_test))
    logger.info("Memory size for test data: " + str(sentence_size_test))
    """


def simple_lstm():
    config_options = globals.config
    input_dim = int(config_options.get('Train', 'input-dim'))
    model = Sequential()
    model.add(LSTM(64, input_shape=(input_dim, 300),))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

def bidirectional_lstm():
    config_options = globals.config
    input_dim = int(config_options.get('Train', 'input-dim'))
    left = Sequential()
    left.add(LSTM(output_dim=32,
                  init='uniform',
                  inner_init='uniform',
                  forget_bias_init='one',
                  return_sequences=False,
                  activation='tanh',
                  inner_activation='sigmoid',
                  input_shape=(input_dim, 300)))
    right = Sequential()
    right.add(LSTM(output_dim=32,
                   init='uniform',
                   inner_init='uniform',
                   forget_bias_init='one',
                   return_sequences=False,
                   activation='tanh',
                   inner_activation='sigmoid',
                   input_shape=(input_dim, 300),
                   go_backwards=True))
    model = Sequential()
    model.add(Merge([left, right], mode='sum'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

def memory_network():
    vocab = Alphabet.from_iterable(word for sent in X for word in sent)
    vocab_dim = 300 # dimensionality of your word vectors
    n_symbols = len(vocab) + 1 # adding 1 to account for 0th index (for masking)
    embedding_weights = np.zeros((n_symbols+1, vocab_dim))
    for word,index in vocab._mapping.items():
        vector = modules.w2v.transform(word)
        if vector is not None:
            embedding_weights[index+1,:] = vector
    #X = [np.array([vocab[word]+1 for word in sent]) for sent in X]
    Xtrain = []
    for sent in X:
        line = np.array([vocab[word]+1 for word in sent])
        Xtrain.append(line)
    X = np.array(Xtrain)
    Y = np.array(Y)
    # assemble the model
    model = Sequential() # or Graph or whatever
    model.add(
        Embedding(output_dim=300,
                  input_dim=n_symbols + 1,
                  mask_zero=True,
                  weights=[embedding_weights])
    )
    model.add(
        LSTM(32,
             return_sequences=False)
    )
    model.add(
        Dense(1,
              activation='sigmoid')
    )
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy')
    model.fit(X, Y)
    save_model_to_file(model, "modelstruct", "modelweights")

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
    load_data()
    exit()

    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    training_data = config_options.get('Train', 'training-data')
    model_file = config_options.get('Train', 'model-file')

    vocab = codecsLoadJson(vocab_file)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    batch_size = FLAGS.batch_size
    sentence_size = 28
    memory_size = 20
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
        for t in range(1, FLAGS.epochs+1):
            loss = 0
            for i in xrange(len(queries)):
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
    config_options = globals.config
    training_data = config_options.get('Train', 'training-data')
    model_struct = config_options.get('Train', 'model-struct')
    model_weights = config_options.get('Train', 'model-weights')
    input_dim = int(config_options.get('Train', 'input-dim'))

    logger.info("Using training data from path: " + training_data)
    logger.info("Saving model struct to path: " + model_struct)
    logger.info("Saving model weights to path: " + model_weights)

    model = Sequential()
    model.add(Dense(64, input_dim=64))
    model.add(MemoryNetwork(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    X = np.random.random((10000, 64))
    Y = np.ones(10000)

    model.fit(X, Y)
    """

    """
    #model = bidirectional_lstm()
    model = simple_lstm()

    X = []
    Y = []
    batch_size = 10000
    count = 0
    num = 0
    lines = codecsReadFile(training_data).strip().split("\n")
    logger.info("Total " + str(len(lines)) + " training samples.")

    for line in lines:
        vectors, label = process_line(line, input_dim)
        X.append(vectors)
        Y.append(label)

        count += 1

        if (count >= batch_size):
            X = np.array(X)
            Y = np.array(Y)
            model.fit(X, Y)
            X = []
            Y = []
            count = 0
            num += 1
            logger.info("Processing batch number " + str(num))

    if X != []:
        X = np.array(X)
        Y = np.array(Y)
        model.fit(X, Y)

    save_model_to_file(model, model_struct, model_weights)
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
    predictions = model.predict_proba(S, Q)

    if n == 1:
        idx = np.argmax(predictions[0])
        return indices[idx]

    new_index = []
    ss = []
    for i in xrange(n):
        arr = predictions[i]
        best = np.argmax(arr)
        ss.append(S[i][best])
        best = i * memory_size + best
        new_index.append(indices[best])

    testS = []
    testQ = []
    nn = len(ss) / memory_size + 1
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
    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    testing_data = config_options.get('Test', 'testing-data')
    model_file = config_options.get('Train', 'model-file')
    test_result = config_options.get('Test', 'test-result')

    vocab = codecsLoadJson(vocab_file)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    batch_size = FLAGS.batch_size
    sentence_size = 28
    memory_size = 20
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
        correct = 0
        for i in xrange(len(queries)):
            query = queries[i]
            #logger.info("Processing question " + str(query.id))
            d = load_data_from_disk(query, testing_data)
            data.append(d)
            if len(data) >= 100:
                trainS, trainQ, trainA = selective_data(data, word_idx, sentence_size, memory_size)
                trainS, trainQ, trainA = randomize_input(trainS, trainQ, trainA)
                logger.info("Done extracting vectors.")
                predictions = model.predict_proba(trainS, trainQ)
                for j in xrange(len(predictions)):
                    arr = predictions[j]
                    idx = np.argmax(arr)
                    print(idx)
                    indicator = trainA[j][idx]
                    if indicator == 1.0:
                        correct += 1
                logger.info("Current accuracy = " + str(float(correct) / i))
                data = []
        logger.info("Accuracy = " + str(float(correct) / len(queries)))

        """
        queries = load_eval_queries(dataset)
        codecsWriteFile(test_result, "")
        correct = 0
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

    """
    config_options = globals.config
    model_struct = config_options.get('Train', 'model-struct')
    model_weights = config_options.get('Train', 'model-weights')
    test_result = config_options.get('Test', 'test-result')
    input_dim = int(config_options.get('Train', 'input-dim'))

    model = load_model(model_struct, model_weights)
    queries = load_eval_queries(dataset)
    codecsWriteFile(test_result, "")
    for query in queries:
        facts = modules.extractor.extract_fact_list_with_entity_linker(query)

        question = query.utterance.lower()
        logger.info("Testing question " + question)
        logger.info("Processing question " + str(query.id))
        tokens = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in question[:-1].split()]
        answer = query.target_result

        input_facts = []
        for fact in facts:
            sid, s, r, oid, o = fact
            if not o.startswith("g."):
                objects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in o.split()]
                if (len(objects) <= 10):
                    input_facts.append(fact)
    """

    """
        inputs = []
        total_scores = None
        count = 0
        for fact in input_facts:
            sid, s, r, oid, o = fact
            relations = re.split("\.\.|\.", r)[:-2]
            rels = [e for t in relations for e in re.split('\.\.|\.|_', t)]
            subjects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in s.split()]
            objects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in o.split()]

            sentence = tokens + subjects + rels + objects
            input_vector = transform_to_vectors(sentence, input_dim)
            inputs.append(input_vector)
            count += 1

            if len(inputs) >= 32:
                inputs = np.array(inputs)
                scores = model.predict(inputs)
                if (total_scores is None):
                    total_scores = scores
                else:
                    total_scores = np.concatenate([total_scores, scores])
                inputs = []

        if count == 0:
            result_line = "\t".join([str(query.id) + question, str(answer), str([])]) + "\n"
            codecsWriteFile(test_result, result_line, "a")
            continue

        if inputs != []:
            inputs = np.array(inputs)
            scores = model.predict(inputs)
            if (total_scores is None):
                total_scores = scores
            else:
                total_scores = np.concatenate([total_scores, scores])

        predictions = []
        assert(len(total_scores) == len(input_facts))
        idx = total_scores.argmax()
        _, best_s, best_r, _, best_o = input_facts[idx]
        for i in xrange(len(total_scores)):
            sid, s, r, oid, o = input_facts[i]
            if best_s == s and best_r == r:
                predictions.append(o)
    """

    """
        if input_facts:
            predictions = memory_network_computation(tokens, input_facts)
        else:
            predictions = []
        result_line = "\t".join([str(query.id) + question, str(answer), str(predictions)]) + "\n"
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