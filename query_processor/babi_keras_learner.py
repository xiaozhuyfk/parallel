"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data, bAbI_transform_data
from sklearn import cross_validation
from itertools import chain

import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM
from functools import reduce


tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "../training_data/bAbI_task/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
S, Q, A = bAbI_transform_data(train, word_idx, sentence_size, memory_size)
trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=FLAGS.random_state)
testS, testQ, testA = bAbI_transform_data(test, word_idx, sentence_size, memory_size)


# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=64))
input_encoder_m.add(Dropout(0.3))
# output: (samples, story_maxlen, embedding_dim)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64))
question_encoder.add(Dropout(0.3))
# output: (samples, query_maxlen, embedding_dim)

# compute a 'match' between input sequence elements (which are vectors)
# and the question vector sequence
match = Sequential()
match.add(Merge([input_encoder_m, question_encoder],
                mode='dot',
                dot_axes=[2, 2]))
match.add(Activation('softmax'))
# output: (samples, story_maxlen, query_maxlen)

# embed the input into a single vector with size = story_maxlen:
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=sentence_size))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

# sum the match vector with the input vector:
response = Sequential()
response.add(Merge([match, input_encoder_c], mode='sum'))
# output: (samples, story_maxlen, query_maxlen)

response.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)

# concatenate the match vector with the question vector,
# and do logistic regression on top
answer = Sequential()
answer.add(Merge([response, question_encoder], mode='concat', concat_axis=-1))
# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.
answer.add(LSTM(32))
# one regularization layer -- more would probably be needed.
answer.add(Dropout(0.3))
answer.add(Dense(memory_size))
# we output a probability distribution over the vocabulary
answer.add(Activation('softmax'))

answer.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])
# Note: you could use a Graph model to avoid repeat the input twice
answer.fit([trainS, trainQ, trainS], trainA,
           batch_size=32,
           nb_epoch=120,
           validation_data=([valS, valQ, valS], valQ))
