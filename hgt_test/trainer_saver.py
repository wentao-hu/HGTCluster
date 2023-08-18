"""Distributed trainers on TensorFlow backend"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time

import graphlearn as gl
import graphlearn.python.nn.tf as tfg
import numpy as np
import tensorflow as tf


class DistTrainer(object):
    """Class for distributed training and evaluation

    Args:
      cluster_spec: TensorFlow ClusterSpec.
      job_name: name of this worker.
      task_index: index of this worker.
      worker_count: The number of TensorFlow worker.
      ckpt_dir: checkpoint dir.
    """

    def __init__(self,
                 cluster_spec,
                 job_name,
                 task_index,
                 worker_count,
                 ckpt_dir=None, ckpt_freq=36000, ckpt_steps=None, max_training_step=None):
        self.cluster_spec = cluster_spec
        self.job_name = job_name
        self.task_index = task_index
        self.worker_count = worker_count
        # TODO
        self.ckpt_dir = ckpt_dir
        self.ckpt_freq = ckpt_freq
        # self.ckpt_steps = ckpt_steps
        self.max_training_step = max_training_step

        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        conf.allow_soft_placement = False
        conf.device_filters.append('/job:ps')
        conf.device_filters.append('/job:worker/task:%d' % self.task_index)
        # conf.inter_op_parallelism_threads = 1
        self.server = tf.train.Server(self.cluster_spec,
                                      job_name=self.job_name,
                                      task_index=self.task_index,
                                      config=conf)
        self.context = self.context
        self.sync_barrier = tfg.SyncBarrierHook(self.worker_count, self.task_index == 0)

    def __exit__(self, exc_type, exc_value, tracebac):
        if self.sess:
            self.sess.close()
        return True

    def close(self):
        if self.sess:
            self.sess.close()
        return True

    def context(self):
        return tf.device(tf.train.replica_device_setter(
            worker_device='/job:worker/task:%d' % self.task_index,
            cluster=self.cluster_spec))

    def init_session(self):
        hooks = [self.sync_barrier]
        # hooks = []
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        conf.allow_soft_placement = False
        conf.device_filters.append('/job:ps')
        conf.device_filters.append('/job:worker/task:%d' % self.task_index)
        # conf.inter_op_parallelism_threads = 1
        if self.ckpt_dir is not None:
            self.sess = tf.train.MonitoredTrainingSession(
                master=self.server.target,
                checkpoint_dir=self.ckpt_dir,
                save_checkpoint_secs=self.ckpt_freq,
                # save_checkpoint_steps=self.ckpt_steps,
                is_chief=(self.task_index == 0),
                hooks=hooks,
                config=conf)
        else:
            self.sess = tf.train.MonitoredTrainingSession(
                master=self.server.target,
                is_chief=(self.task_index == 0),
                hooks=hooks,
                config=conf)

    def _train_one_epoch(self,
                        sess,
                        epoch,
                        train_ops):
        local_step = 0
        t = time.time()
        global_step = sess._tf_sess().run(tf.train.get_or_create_global_step())
        last_global_step = global_step
        while not self.sess.should_stop():
            # early quit
            if self.max_training_step is not None and local_step > self.max_training_step:
                print('Early quit epoch since reaching max training step {} {}'.format(
                    local_step, self.max_training_step))
                break
            try:
                outs = self.sess.run(train_ops)
            except tf.errors.OutOfRangeError:
                print('End of an epoch {}.'.format(epoch))
                break
            train_loss = outs[1]
            global_step = outs[2]
            acc = outs[3]
            # Print results
            if local_step <= 20 or local_step % 10 == 0:
                print(datetime.datetime.now(),
                      'Epoch {}, Iter {}, global_step {}, Global_step/sec {:.2f}, Time(s) {:.4f}, '
                      'Loss {:.5f}, Acc {:.5f}'
                      .format(epoch, local_step, global_step,
                              (global_step - last_global_step) * 1.0 / (time.time() - t),
                              (time.time() - t) * 1.0 / 10,
                              train_loss, acc))
                t = time.time()
                last_global_step = global_step
            local_step += 1

        return global_step

    def _save_one_model(self, sess, saver, global_step=None):
        global_step = global_step or sess._tf_sess().run(tf.train.get_or_create_global_step())
        print("Start saving checkpoint in {} global_step = {}...".format(self.ckpt_dir, global_step))
        saver.save(sess._tf_sess(), "{}/ckpt/model".format(self.ckpt_dir), global_step=global_step)


    def train(self, iterator_list, loss, optimizer=None, learning_rate=None, epochs=10, metrics=[], **kwargs):
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
            train_ops = [train_op, loss,
                         self.global_step] + metrics
            # if self.task_index == 0:
            #     saver = tf.train.Saver(sharded=True, allow_empty=True)
            self.init_session()
            print('Start training...')
            # for iterator in iterator_list:
            #     iterator.make_one_shot_iterator()
            for epoch in range(epochs):
                print("=" * 20)
                print("Start on epoch {}".format(epoch))
                print("=" * 20)
                print("Initialize dataset")
                # initailize dataset
                self.sess._tf_sess().run([iterator.initializer for iterator in iterator_list])
                # train on an epoch
                print("Start train on epoch {}".format(epoch))
                global_step = self._train_one_epoch(self.sess, epoch, train_ops)
                print("Finsh train on epoch {} global_step = {}".format(epoch, self.sess._tf_sess().run(tf.train.get_or_create_global_step())))
                # save model
                # TODO
                # print("Save checkpoint for epoch {} global_step = {}".format(epoch, global_step))
                # if self.task_index == 0:
                #     self._save_one_model(self.sess, saver, global_step)

            self.sync_barrier.end(self.sess)

    def save_node_embedding(self, emb_path, iterator, ids, emb, batch_size):
        print('Start saving embeddings...')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        with self.context():
            self.global_step = tf.train.get_or_create_global_step()
            if self.sess is None:
                self.init_session()
            local_step = 0
            self.sess._tf_sess().run(iterator.initializer)
            emb_dict = {}
            while True:
                try:
                    t = time.time()
                    outs = self.sess._tf_sess().run([ids, emb])
                    # [B,], [B,dim]
                    feat = [','.join(str(x) for x in arr) for arr in outs[1]]
                    res = list(zip(outs[0], feat))  # id,emb
                    for (id, value) in res:
                        emb_dict[id] = value
                    local_step += 1
                except tf.errors.OutOfRangeError:
                    print("Start write to file...")
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    with open(emb_path, 'w') as f:
                        for id in emb_dict:
                            f.write("{}\t{}\n".format(id, emb_dict[id]))
                    print('Save node embeddings done.')
                    break

    def join(self):
        self.server.join()
