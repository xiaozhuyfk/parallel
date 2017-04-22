from __future__ import print_function

import logging
import globals
import modules
from evaluation import load_eval_queries
from util import codecsWriteFile, codecsReadFile, codecsLoadJson
import re
from keras.layers import Input, LSTM, Dense, Embedding, Merge, Bidirectional
from keras.models import Model, model_from_json, Sequential
import numpy as np
import random
from alphabet import Alphabet
import os
import tensorflow as tf
from memory_network import MemN2N

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

def load_data_from_disk(query, path):
        id = query.id
        file_path = path + str(id)
        if os.path.isfile(file_path):
            d = codecsLoadJson(file_path)
            return d
        else:
            return None

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

    model = Sequential()
    model.add(Bidirectional(
        LSTM(output_dim=32, input_shape=(input_dim, 300))
    ))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

def bidirectional_lstm_with_embedding(vocab_dim, n_symbols, word_idx):
    logger.info("Initializing embedding weights.")
    embedding_weights = np.zeros((n_symbols+1, vocab_dim))
    for word, index in word_idx.items():
        vector = modules.w2v.transform(word)
        if vector is not None:
            embedding_weights[index,:] = vector
        else:
            embedding_weights[index,:] = np.random.normal(0, 0.1, vocab_dim)

    # assemble the model
    logger.info("Constructing Bi-directional LSTM model.")
    model = Sequential()

    model.add(
        Embedding(output_dim=vocab_dim,
                  input_dim=n_symbols+1,
                  mask_zero=True,
                  weights=[embedding_weights])
    )

    """
    model.add(
        Embedding(output_dim=vocab_dim,
                  input_dim=n_symbols,
                  mask_zero=True)
    )
    """

    model.add(
        Bidirectional(LSTM(32))
    )
    model.add(
        Dense(1)
    )
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy')

    return model

def selective_data(data, word_idx, sentence_size, memory_size):
    S = []
    A = []
    for d in data:
        query = d.get("query")
        story = d.get("story")

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        correct = []
        wrong = []
        for i in xrange(len(ss)):
            indicator = d.get("y")[i]
            if (indicator == 1.0):
                correct.append(ss[i])
            else:
                wrong.append(ss[i])

        if len(correct) > 0:
            if len(correct) >= memory_size:
                s = correct[:memory_size]
                a = [1.0] * memory_size
            else:
                if len(wrong) >= memory_size - len(correct):
                    s = correct + random.sample(wrong, memory_size - len(correct))
                    a = [1.0] * len(correct) + [0.0] * (memory_size - len(correct))
                else:
                    lm = memory_size - len(correct) - len(wrong)
                    s = correct + wrong
                    a = [1.0] * len(correct) + [0.0] * len(wrong)
                    for _ in xrange(lm):
                        s.append([0] * sentence_size)
                        a.append(0.0)
        else:
            if (len(wrong) >= memory_size):
                s = random.sample(wrong, memory_size)
                a = [0.0] * memory_size
            else:
                lm = memory_size - len(wrong)
                for _ in xrange(lm):
                    wrong.append([0] * sentence_size)
                s = wrong
                a = [0.0] * memory_size

        shuffle_s = []
        shuffle_a = []
        indices = range(len(s))
        random.shuffle(indices)
        for i in indices:
            shuffle_s.append(s[i] + q)
            shuffle_a.append(a[i])
        S += shuffle_s
        A += shuffle_a

    return np.array(S), np.array(A)

def generate_data(dataset):
    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    training_data = config_options.get('Train', 'training-data')

    vocab = codecsLoadJson(vocab_file)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    queries = load_eval_queries(dataset)
    sentence_size = 17
    memory_size = 10
    while True:
        data = []
        for i in xrange(len(queries)):
            query = queries[i]
            d = load_data_from_disk(query, training_data)
            data.append(d)
            if len(data) >= 100:
                X, Y = selective_data(data, word_idx, sentence_size, memory_size)
                yield (X, Y)
                data = []

        if len(data) > 0:
            X, Y = selective_data(data, word_idx, sentence_size, memory_size)
            yield (X, Y)

def train(dataset):
    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    training_data = config_options.get('Train', 'training-data')
    model_struct = config_options.get('Train', 'model-struct')
    model_weights = config_options.get('Train', 'model-weights')

    vocab = codecsLoadJson(vocab_file)
    n_symbols = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    sentence_size = 17
    memory_size = 10
    model = bidirectional_lstm_with_embedding(300, n_symbols, word_idx)

    model.fit_generator(generate_data(dataset),
                        samples_per_epoch=37780,
                        nb_epoch=10)
    """
    queries = load_eval_queries(dataset)
    data = []
    for i in xrange(len(queries)):
        query = queries[i]
        logger.info("Processing question " + str(query.id))

        d = load_data_from_disk(query, training_data)
        data.append(d)

        if len(data) >= 100:
            X, Y = selective_data(data, word_idx, sentence_size, memory_size)
            model.fit(X, Y)
            data = []

    if len(data) > 0:
        X, Y = selective_data(data, word_idx, sentence_size, memory_size)
        model.fit(X, Y)
    """

    save_model_to_file(model, model_struct, model_weights)

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

def memory_data(data, word_idx, sentence_size, memory_size):
    S = []
    A = []
    for d in data:
        query = d.get("query")
        story = d.get("story")
        A = d.get("y")

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            S.append([word_idx[w] for w in sentence] + [0] * ls + q)

    return np.array(S), np.array(A)

def test(dataset):
    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    model_struct = config_options.get('Train', 'model-struct')
    model_weights = config_options.get('Train', 'model-weights')
    testing_data = config_options.get('Test', 'testing-data')
    test_result = config_options.get('Test', 'test-result')

    vocab = codecsLoadJson(vocab_file)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    sentence_size = 28
    memory_size = 20
    model = load_model(model_struct, model_weights)

    """
    queries = load_eval_queries(dataset)
    data = []
    correct = 0
    for i in xrange(len(queries)):
        query = queries[i]
        logger.info("Processing question " + str(query.id))
        d = load_data_from_disk(query, testing_data)
        data.append(d)
        if len(data) >= 100:
            X, Y = selective_data(data, word_idx, sentence_size, memory_size)
            predictions = model.predict(X)
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
    for i in xrange(len(queries)):
        query = queries[i]
        question = query.utterance.lower()
        answer = query.target_result

        logger.info("Processing question " + str(query.id))
        d = load_data_from_disk(query, testing_data)

        y = d.get("y")
        if len(y) == 0:
            result_line = "\t".join([str(query.id) + question, str(answer), str([])]) + "\n"
            codecsWriteFile(test_result, result_line, "a")
            continue

        result = []
        for i in xrange(len(y)):
            if y[i] == 1.0:
                result.append(" ".join(d.get("O")[i]))
        result_line = "\t".join([str(query.id) + question, str(answer), str(result)]) + "\n"
        codecsWriteFile(test_result, result_line, "a")

        """
        X, Y = memory_data([d], word_idx, sentence_size, memory_size)
        predictions = model.predict(X)
        if len(predictions) == 0:
            result_line = "\t".join([str(query.id) + question, str(answer), str([])]) + "\n"
            codecsWriteFile(test_result, result_line, "a")
            continue

        idx = np.argmax(predictions)

        best_s = " ".join(d.get("S")[idx])
        best_r = " ".join(d.get("R")[idx])
        result = []

        for i in xrange(len(d.get("O"))):
            s = " ".join(d.get("S")[i])
            r = " ".join(d.get("R")[i])

            if s == best_s and r == best_r:
                result.append(" ".join(d.get("O")[i]))
        result_line = "\t".join([str(query.id) + question, str(answer), str(result)]) + "\n"
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