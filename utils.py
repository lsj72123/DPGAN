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

"""This file contains some utility functions"""

import tensorflow as tf
import time
import os

FLAGS = tf.app.flags.FLAGS


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def load_ckpt(saver, sess, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess,
    waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir == "eval" else None
            ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
            time.sleep(10)


def setup_model(model, mode):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, mode)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    model.build_graph()  # build the graph

    saver = tf.train.Saver(max_to_keep=10)  # we use this to load checkpoints for decoding
    sess = tf.Session(config=get_config())
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess, saver, train_dir


def run_pre_train_generator(model, batcher, max_run_epoch, sess, saver, train_dir):
    tf.logging.info("starting run_pre_train_generator")
    epoch = 0
    while epoch < max_run_epoch:
        batches = batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            # print_batch(current_batch)
            step += 1
            results = model.run_pre_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training generator step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0
            if train_step % 100 == 0:
                saver.save(sess, train_dir + "/model", global_step=train_step)
                # bleu_score = generated.compute_BLEU(str(train_step))
                # tf.logging.info('bleu: %f', bleu_score)  # print the loss to screen

        epoch += 1
        tf.logging.info("finished %d epoches", epoch)
