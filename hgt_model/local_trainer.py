# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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
# =============================================================================
"""Distributed trainers on TensorFlow backend"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import contextlib
import datetime
import os
import time

import numpy as np
import graphlearn as gl
import graphlearn.python.nn.tf as tfg

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from tensorflow.python.client import timeline
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

class TFTrainer(object):
  """Class for local or distributed training and evaluation.

  Args:
    ckpt_dir: checkpoint dir.
    save_checkpoint_secs: checkpoint frequency.
    save_checkpoint_steps: checkpoint steps.
    profiling: whether write timeline for profiling, default is False.
    progress_steps: print a progress logs for given steps.
  """
  def __init__(self,
               ckpt_dir=None,
               save_checkpoint_secs=600,
               save_checkpoint_steps=None,
               profiling=False,
               progress_steps=10):
    self.ckpt_dir = ckpt_dir
    self.save_checkpoint_secs = save_checkpoint_secs
    self.save_checkpoint_steps = save_checkpoint_steps
    self.profiling = profiling
    self.progress_steps = progress_steps

    self.conf = tf.ConfigProto()
    self.conf.gpu_options.allow_growth = True
    self.conf.allow_soft_placement = False
    self.sess = None

    # use for distributed training
    self.sync_barrier = None
    self.global_step = None
    self.is_local = None

  def context(self):
    raise NotImplementedError('Use LocalTrainer or DistTrainer instead.')

  def init_session(self, hooks=None, **kwargs):
    if isinstance(hooks, (list, tuple)):
      hooks_ = [hook for hook in hooks]
    elif hooks is not None:
      hooks_ = [hooks]

    checkpoint_args = dict()
    if self.ckpt_dir is not None:
      checkpoint_args['checkpoint_dir'] = self.ckpt_dir
    if self.save_checkpoint_secs is not None:
      checkpoint_args['save_checkpoint_secs'] = self.save_checkpoint_secs
    if self.save_checkpoint_steps is not None:
      checkpoint_args['save_checkpoint_steps'] = self.save_checkpoint_steps

    self.sess = tf.train.MonitoredTrainingSession(
        hooks=hooks_,
        config=self.conf,
        **checkpoint_args,
        **kwargs)

    def _close_session():
      if self.sess is not None:
        self.sess.close()
    atexit.register(_close_session)

  def run_step(self, train_ops, local_step):
    raise NotImplementedError('Use LocalTrainer or DistTrainer instead.')

  def train(self, iterator, loss, optimizer=None, learning_rate=None,
            epochs=10, hooks=[], metrics=[], **kwargs):
    with self.context():
      self.global_step = tf.train.get_or_create_global_step()
      if optimizer is None:
        try:
          optimizer = tf.train.AdamAsyncOptimizer(learning_rate=learning_rate)
        except AttributeError:
          optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = optimizer.minimize(loss, global_step=self.global_step)
      # if self._use_input_bn:
      #   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      #   train_op = tf.group([train_op, update_ops])
      train_ops = [train_op, loss, self.global_step] + metrics
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
      self.init_session(hooks=hooks)

      print('Start training...')
      local_step = 0
      last_local_step = 0
      last_global_step = 0
      t = time.time()
      epoch = 0
      outs = None
      while (not self.sess.should_stop()) and (epoch < epochs):
        try:
          outs = self.run_step(train_ops, local_step)
        except tf.errors.OutOfRangeError:
          print('End of the epoch %d.' % (epoch,))
          epoch += 1
          self.sess._tf_sess().run(iterator.initializer)  # reinitialize dataset.
        if outs is not None:
          train_loss = outs[1]
          global_step = outs[2]
          if len(metrics) > 0:
            acc= outs[3]
          else:
            acc = 0.0

          # Print results
          local_step += 1
          if local_step % self.progress_steps == 0:
            if self.is_local:
              print(datetime.datetime.now(),
                    'Epoch {}, Iter {}, LocalStep/sec {:.2f}, Time(s) {:.4f}, '
                    'Loss {:.5f}, Accuracy {:.5f}'
                    .format(epoch, local_step,
                            (local_step - last_local_step) * 1.0 / (time.time() - t),
                            (time.time() - t) * 1.0 / 10, train_loss, acc))
            else:
              print(datetime.datetime.now(),
                    'Epoch {}, Iter {}, GlobalStep/sec {:.2f}, Time(s) {:.4f}, '
                    'Loss {:.5f}, Global_step {}, Accuracy {:.5f}'
                    .format(epoch, local_step,
                            (global_step - last_global_step) * 1.0 / (time.time() - t),
                            (time.time() - t) * 1.0 / 10, train_loss, global_step, acc))
            t = time.time()
            last_local_step = local_step
            last_global_step = global_step

      if self.sync_barrier is not None:
        self.sync_barrier.end(self.sess)



  def train_and_evaluate(self, train_iterator, test_iterator, loss, test_acc, optimizer=None, learning_rate=None,
                         epochs=10, hooks=[], **kwargs):
    self.train(train_iterator, loss, optimizer, learning_rate, epochs, hooks, **kwargs)
    self.test(test_iterator, test_acc, hooks, **kwargs)

  def save_node_embedding(self, emb_writer, iterator, ids, emb, batch_size):
    print('Start saving embeddings...')
    with self.context():
      self.global_step = tf.train.get_or_create_global_step()
      if self.sess is None:
        self.init_session()
      local_step = 0
      self.sess._tf_sess().run(iterator.initializer)
      while True:
        try:
          t = time.time()
          outs = self.sess._tf_sess().run([ids, emb])
          # [B,], [B,dim]
          feat = [','.join(str(x) for x in arr) for arr in outs[1]]
          emb_writer.write(list(zip(outs[0], feat)), indices=[0, 1])  # id,emb
          local_step += 1
          if local_step % self.progress_steps == 0:
            print('Saved {} node embeddings, Time(s) {:.4f}'.format(local_step * batch_size, time.time() - t))
        except tf.errors.OutOfRangeError:
          print('Save node embeddings done.')
          break

class LocalTrainer(TFTrainer):
  """Class for local training and evaluation

  Args:
    ckpt_dir: checkpoint dir.
    save_checkpoint_freq: checkpoint frequency.
    save_checkpoint_steps: checkpoint steps.
    profiling: whether write timeline for profiling, default is False.
    progress_steps: print a progress logs for given steps.
  """
  def __init__(self,
               ckpt_dir=None,
               save_checkpoint_secs=None,
               save_checkpoint_steps=None,
               profiling=False,
               progress_steps=10):
    super().__init__(ckpt_dir, save_checkpoint_secs, save_checkpoint_steps, profiling, progress_steps)
    self.is_local = True 

  if hasattr(contextlib, 'nullcontext'):
    def context(self):
      return contextlib.nullcontext()
  else:
    @contextlib.contextmanager
    def context(self, enter_result=None):
        yield enter_result

  def run_step(self, train_ops, local_step):
    if self.profiling and local_step % 100 == 0 and local_step > 500 and local_step < 1000:
      outs = self.sess.run(train_ops,
                           options=run_options,
                           run_metadata=run_metadata)
      tl = timeline.Timeline(run_metadata.step_stats)
      content = tl.generate_chrome_trace_format()
      file_name = 'timeline_' + str(local_step) + '.json'
      save_path = os.path.join(self.ckpt_dir, file_name)
      writeGFile = tf.gfile.GFile(save_path, mode='w')
      writeGFile.write(content)
      writeGFile.flush()
      writeGFile.close()
      print("Profiling data save to %s success." % save_path)
    else:
      outs = self.sess.run(train_ops)
    return outs

