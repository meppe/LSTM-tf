# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import system
import os
from tensorflow.python.platform import gfile

import time

import numpy as np
import tensorflow as tf

from reader import Reader

# This is used for Python Debug Configuration only
# import pydevd
# server_ip = '192.168.4.232'
# server_port=51234
# pydevd.settrace(server_ip, port=server_port, stdoutToServer=True, stderrToServer=True)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("dev_data", True,
                  "Train using toy data files")

flags.DEFINE_string("working_path", os.getcwd(), "working_path")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        target_size = num_steps
        # target_size = 1 # Only check the next symbol or word.

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, target_size])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b

        self.prediction = logits
        self.reshaped_targets = tf.reshape(self._targets, [-1])

        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class DevConfig(object):
    """Tiny toy config, for development."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 1
    num_steps = 3
    hidden_size = 12
    max_epoch = 2
    max_max_epoch = 5
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    vocab_size = 12


def run_epoch(session, model, data, eval_op, reader, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                      model.num_steps)):
        fetches = [model.reshaped_targets, model.prediction, model.cost, model.final_state, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
            reshaped_targets, prediction, cost, state, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.num_steps

        if verbose and epoch_size > 10 and step % (epoch_size // 10) == 10:
            print("%.1f perplexity: %.5f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    elif FLAGS.model == "dev":
        return DevConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

def get_next_symbols(reader, past_symbols, model, session):
    past_sym_ids = []
    for s in past_symbols:
      past_sym_ids += [reader.word_to_id_table[s]]

    state = session.run(model.initial_state)

    fetches = [model.reshaped_targets, model.prediction, model.cost, model.final_state, tf.no_op()]
    feed_dict = {}
    feed_dict[model.input_data] = [past_sym_ids]
    feed_dict[model.targets] = [past_sym_ids]
    for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h
        reshaped_targets, prediction, cost, state, _ = session.run(fetches, feed_dict)

    next_sym_ids = []
    for prop in prediction:
        next_sym_ids += [np.argmax(prop)]

    next_syms = []
    for si in next_sym_ids:
        next_syms += [reader.id_to_word_table[si]]
    return next_syms

def main(_):

    # Training pipeline

    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    r = Reader()

    raw_data = r.ptb_raw_data(FLAGS.data_path, FLAGS.dev_data)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)


        saver = tf.train.Saver()

        print("Do you want to restore the last model (y/n)?")
        yn = raw_input()
        if yn == "y":
            saver.restore(session, FLAGS.working_path+"/last_model")
        else:
            init_op = tf.initialize_all_variables()
            init_op.run()

            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.8f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, train_data, m.train_op, r,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.8f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op(), r)
                print("Epoch: %d Valid Perplexity: %.8f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest, test_data, tf.no_op(), r)
            print("Test Perplexity: %.8f" % test_perplexity)

            model_path = FLAGS.working_path+"/model_"+str(time.strftime("%d_%b_%Y_%H:%M:%S", time.localtime()))
            save_path = saver.save(session, model_path)
            print("Model saved in path: %s" % save_path)
            system("rm "+model_path)
            system("ln -s "+model_path+" last_model")

        # Text generation pipeline:
        # Decide which model to use as generation model
        gmodel = mvalid

        print("Enter {} of the following words or characters to start with, separated by space".format(gmodel.num_steps))
        symbols_unique = r.id_to_word_table.values()
        print(symbols_unique)
        next_syms = []
        while len(next_syms) < gmodel.num_steps:
            in_chrs = raw_input()
            in_chrs = in_chrs.split(" ")
            for in_chr in in_chrs:
                if in_chr not in symbols_unique:
                    print("invalid symbol '{}', try again".format(in_chr))
                    continue
            if len(in_chrs) != gmodel.num_steps:
                print("invalid number of symbols, try again")
                continue

            next_syms = in_chrs

        print("Generating text. Press <Enter> for next word, or enter 'q' to quit")
        while True:
            next_syms = get_next_symbols(r, next_syms, gmodel, session)
            for sym in next_syms:
                print(sym + " ")

            in_chrs = raw_input()
            if in_chrs == "q":
                break
            elif in_chrs != "":
                in_chrs = in_chrs.split(" ")
                valid = True
                for in_chr in in_chrs:
                    if in_chr not in symbols_unique:
                        print("invalid symbol, using generated symbols")
                        valid = False
                        break
                if len(in_chrs) != gmodel.num_steps:
                    valid = False
                if valid:
                    next_syms = in_chrs

        print("Finished all -- End!")


if __name__ == "__main__":
    tf.app.run()
