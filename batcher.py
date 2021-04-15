# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
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

"""This file contains code to process data into batches"""

import numpy as np
import tensorflow as tf
import data
import codecs
import json
import os
import random
import pickle
import copy
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from glob import glob

FLAGS = tf.app.flags.FLAGS


class Example(object):
    """Class representing a train/val/test example for text summarization."""

    def __init__(self, review, vocab, hps, article=None):
        """Initializes the Example, performing tokenization and truncation to produce the
        encoder, decoder and target sequences, which are stored in self.

        Args:
           review: source text; a string. each token is separated by a single space.
           vocab: list of strings, one per abstract sentence.
                In each sentence, each token is separated by a single space.
           hps: Vocabulary object
           article: hyperparameters
        """
        self.original_review = review

        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)
        stop_doc = vocab.word2id(data.STOP_DECODING_DOCUMENT)

        review_sentence = sent_tokenize(review)
        if article is None:
            article = review_sentence[0]
            review_sentence = review_sentence[1:]
            review = " ".join(review_sentence)
        article_words = article.split()
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]
        self.enc_len = len(article_words)
        self.enc_input = [vocab.word2id(w) for w in article_words]
        self.original_review_input = article
        self.original_review_output = review

        abstract_sentences = [x.strip() for x in review_sentence]
        abstract_words = []
        for i, abstract_sen in enumerate(abstract_sentences):
            if i >= hps.max_dec_sen_num:
                break
            abstract_sen_words = abstract_sen.split()
            if len(abstract_sen_words) > hps.max_dec_steps:
                abstract_sen_words = abstract_sen_words[:hps.max_dec_steps]
            abstract_words.append(abstract_sen_words)
            if len(abstract_words[-1]) < hps.max_dec_steps:
                abstract_words[-1].append(stop_doc)

        abs_ids = [[vocab.word2id(w) for w in sen] for sen in abstract_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(
            abs_ids, hps.max_dec_sen_num, hps.max_dec_steps, start_decoding, stop_decoding)

        self.dec_len = len(self.dec_input)
        self.dec_sen_len = [len(sentence) for sentence in self.target]

    @staticmethod
    def get_dec_inp_targ_seqs(sequence, max_sen_num, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input sequence for the decoder,
         and the target sequence which we will use to calculate loss.
         The sequence will be truncated if it is longer than max_len.
         The input sequence must start with the start_id and the target sequence must end with the stop_id
         (but not if it's been truncated).

        Args:
          sequence: List of ids (integers)
          max_sen_num: integer
          max_len: integer
          start_id: integer
          stop_id: integer

        Returns:
          inp: sequence length <=max_len starting with start_id
          target: sequence same length as input, ending with stop_id only if there was no truncation
        """

        inputs = sequence[:max_sen_num] if len(sequence) > max_sen_num else sequence
        targets = copy.deepcopy(inputs)
        for i in range(len(inputs)):
            if len(inputs[i]) >= max_len:
                inputs[i] = inputs[i][:max_len - 1]
                targets[i] = targets[i][:max_len - 1]
            inputs[i] = [start_id] + inputs[i]
            targets[i] = targets[i] + [stop_id]
        return inputs, targets

    def pad_decoder_inp_targ(self, max_sen_len, max_sen_num, pad_doc_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""

        while len(self.dec_sen_len) < max_sen_num:
            self.dec_sen_len.append(1)

        for i in range(len(self.dec_input)):
            while len(self.dec_input[i]) < max_sen_len:
                self.dec_input[i].append(pad_doc_id)

        while len(self.dec_input) < max_sen_num:
            self.dec_input.append([pad_doc_id for _ in range(max_sen_len)])

        for i in range(len(self.target)):
            while len(self.target[i]) < max_sen_len:
                self.target[i].append(pad_doc_id)

        while len(self.target) < max_sen_num:
            self.target.append([pad_doc_id for _ in range(max_sen_len)])

    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)


class Batch(object):
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, hps, vocab):
        """Turns the example_list into a Batch object.

        Args:
           example_list: List of Example objects
           hps: hyperparameters
           vocab: Vocabulary object
        """
        self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        if FLAGS.run_method == 'auto-encoder':
            self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder
        self.init_decoder_seq(example_list, hps)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list, hps):
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros(hps.batch_size, dtype=np.int32)
        # self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i] = ex.enc_input
            self.enc_lens[i] = ex.enc_len

    def init_decoder_seq(self, example_list, hps):
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, hps.max_dec_sen_num, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch
        # because we do not use a dynamic_rnn for decoding.
        # However I believe this is possible, or will soon be possible, with Tensorflow 1.0,
        # in which case it may be best to upgrade to that.
        self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_sen_num, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((hps.batch_size, hps.max_dec_sen_num, hps.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((hps.batch_size * hps.max_dec_sen_num, hps.max_dec_steps), dtype=np.float32)
        self.dec_sen_lens = np.zeros((hps.batch_size, hps.max_dec_sen_num), dtype=np.int32)
        self.dec_lens = np.zeros(hps.batch_size, dtype=np.int32)

        for i, ex in enumerate(example_list):
            self.dec_lens[i] = ex.dec_len
            dec_input = np.asarray(ex.dec_input)
            self.dec_batch[i, :, :] = dec_input
            self.target_batch[i] = np.array(ex.target)
            for j in range(len(ex.dec_sen_len)):
                self.dec_sen_lens[i][j] = ex.dec_sen_len[j]

        self.target_batch = np.reshape(self.target_batch, [hps.batch_size * hps.max_dec_sen_num, hps.max_dec_steps])

        for j in range(len(self.target_batch)):
            for k in range(len(self.target_batch[j])):
                if int(self.target_batch[j][k]) != self.pad_id:
                    self.dec_padding_mask[j][k] = 1

    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""

        self.original_review_output = [ex.original_review_output for ex in example_list]  # list of lists
        if FLAGS.run_method == 'auto-encoder':
            self.original_review_inputs = [ex.original_review_input for ex in example_list]  # list of lists


class GenBatcher(object):
    def __init__(self, vocab, hps):
        self._vocab = vocab
        self._hps = hps

        self.train_queue = self.fill_example_queue(os.path.join(FLAGS.data_path, 'train'))
        self.test_queue = self.fill_example_queue(os.path.join(FLAGS.data_path, 'test'))
        self.train_batch = self.create_batch(os.path.join(FLAGS.data_path, 'train'), mode="train")
        self.test_batch = self.create_batch(os.path.join(FLAGS.data_path, 'test'), mode="test")

    def fill_example_queue(self, data_path):
        example_path = os.path.join(data_path, 'Example.pkl')
        if os.path.exists(example_path):
            with open(example_path, 'rb') as f:
                pyload = pickle.loads(f.read())
            print("Load example file from {}".format(example_path))
            return pyload
        else:
            print("No example file found, creating now ...")
            new_queue = []
            filelist = glob(os.path.join(data_path, '*.txt'))
            assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty

            for f in tqdm(filelist, desc='file list', ncols=80):
                reader = codecs.open(f, 'r', 'utf-8')
                while True:
                    string_ = reader.readline()
                    if not string_:
                        break
                    dict_example = json.loads(string_)
                    review = dict_example["review"]
                    if len(sent_tokenize(review)) < 2:
                        continue
                    example = Example(review, self._vocab, self._hps)
                    new_queue.append(example)
            with open(example_path, 'wb') as f:
                f.write(pickle.dumps(new_queue))
            print("Example file saved to {}".format(example_path))
            return new_queue

    def create_batch(self, data_path, mode="train"):
        all_batch = []
        if mode == "train":
            num_batches = int(len(self.train_queue) / self._hps.batch_size)
        elif mode == 'test':
            num_batches = int(len(self.test_queue) / self._hps.batch_size)
        else:
            raise NotImplementedError

        batch_path = os.path.join(data_path, 'batch.pkl')
        if os.path.exists(batch_path):
            with open(batch_path, 'rb') as f:
                pyload = pickle.loads(f.read())
            print("Load batch file from {}".format(batch_path))
            return pyload
        else:
            print("No batch file found, creating now ...")
            for i in range(0, num_batches):
                batch = []
                if mode == 'train':
                    batch += (self.train_queue[i * self._hps.batch_size:(i + 1) * self._hps.batch_size])
                elif mode == 'test':
                    batch += (self.test_queue[i * self._hps.batch_size:(i + 1) * self._hps.batch_size])
                all_batch.append(Batch(batch, self._hps, self._vocab))
            with open(batch_path, 'wb') as f:
                f.write(pickle.dumps(all_batch))
            print("batch file saved to {}".format(batch_path))
            return all_batch

    def get_batches(self, mode="train"):
        if mode == "train":
            random.shuffle(self.train_batch)
            return self.train_batch
        elif mode == 'test':
            return self.test_batch
