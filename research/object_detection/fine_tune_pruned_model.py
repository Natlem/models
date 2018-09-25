import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-9.0/lib64/"
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pickle
import argparse

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
sys.path.append("../slim/")
from object_detection.utils import ops as utils_ops
import functools
import json
from google.protobuf import text_format
from object_detection import exporter
from object_detection import evaluator_custom
from object_detection.protos import pipeline_pb2
from object_detection import trainer
from object_detection.builders import dataset_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util
from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from deployment import model_deploy
from object_detection.trainer import create_input_queue, _create_losses

from tensorflow.contrib.training.python.training import training
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util
import collections

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)

pruning_methods = ["L1", "Oracle", "Entropy"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pruned model WITHOUT fine-tuning")

    parser.add_argument('--model_ckpt', dest='model_ckpt', help='original ckpt of model')
    parser.add_argument('--model_meta', dest='model_meta', help='original meta of model')
    parser.add_argument('--pruned_weights_path', dest="np_wpath", help="pruned_weight_path")
    parser.add_argument('--pruned_shape_path', dest="np_spath", help="pruned_shape_path")
    parser.add_argument('--train_config', dest='train_config', help='original train config')
    parser.add_argument('--train_dir', dest='train_dir', help='train dir for final')
    args = parser.parse_args()
    return args

def fine_tune_pruned_model(pipeline_config, train_dir, pruned_shape, pruned_weights, checkpoint_path, fine_tune =True):

    task = 0

    if task == 0:
        tf.gfile.MakeDirs(train_dir)
    if pipeline_config:
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        if task == 0:
            tf.gfile.Copy(pipeline_config, os.path.join(train_dir, 'pipeline.config'), overwrite=True)
    else:
        print("FAIL")
        sys.exit(1)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']
    train_config.fine_tune_checkpoint_type = 'detection'

    model_c_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True,
        add_summaries=True,
        convDict=pruned_shape)

    def get_next(config):
        return dataset_util.make_initializable_iterator(
            dataset_builder.build(config)).get_next()

    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    create_model_fn = model_c_fn
    create_tensor_dict_fn = create_input_dict_fn
    pruned_graph = tf.Graph()
    with pruned_graph.as_default():
        detection_model = model_c_fn()
        data_augmentation_options = [preprocessor_builder.build(step) for step in
                                     train_config.data_augmentation_options]

        # Build a configuration specifying multi-GPU and multi-replicas.
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=1,
            clone_on_cpu=False,
            replica_id=task,
            num_replicas=worker_replicas,
            num_ps_tasks=ps_tasks,
            worker_job_name=worker_job_name)

        # Place the global step on the device storing the variables.
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        with tf.device(deploy_config.inputs_device()):
            input_queue = create_input_queue(
                train_config.batch_size // 1, create_input_dict_fn,
                train_config.batch_queue_capacity,
                train_config.num_batch_queue_threads,
                train_config.prefetch_queue_capacity, data_augmentation_options)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        global_summaries = set([])

        model_fn = functools.partial(_create_losses,
                                     create_model_fn=model_c_fn,
                                     train_config=train_config)
        clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
        first_clone_scope = clones[0].scope

        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by model_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        with tf.device(deploy_config.optimizer_device()):
            training_optimizer, optimizer_summary_vars = optimizer_builder.build(train_config.optimizer)
            for var in optimizer_summary_vars:
                tf.summary.scalar(var.op.name, var)

        sync_optimizer = None

        # Create ops required to initialize the model from a given checkpoint.
        init_fn = None
        if pruned_weights:
            init_assign_op, init_feed_dict = slim.assign_from_values(pruned_weights)

            def initializer_fn(sess):
                sess.run(init_assign_op, init_feed_dict)

            init_fn = initializer_fn

        with tf.device(deploy_config.optimizer_device()):
            regularization_losses = (None if train_config.add_regularization_loss
                                     else [])
            total_loss, grads_and_vars = model_deploy.optimize_clones(
                clones, training_optimizer,
                regularization_losses=regularization_losses)
            total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

        # Optionally clip gradients
        if train_config.gradient_clipping_by_norm > 0:
            with tf.name_scope('clip_grads'):
                grads_and_vars = slim.learning.clip_gradient_norms(
                    grads_and_vars, train_config.gradient_clipping_by_norm)

            # Create gradient updates.
            grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                              global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops, name='update_barrier')
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')

        # Add summaries.
        for model_var in slim.get_model_variables():
            global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
        for loss_tensor in tf.losses.get_losses():
            global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
        global_summaries.add(
            tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))
        summaries |= global_summaries

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)

        # Save checkpoints regularly.
        keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
        saver = tf.train.Saver(
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
        with pruned_graph.as_default():
            slim.learning.train(
                train_tensor,
                logdir=train_dir,
                master=master,
                is_chief=is_chief,
                session_config=session_config,
                startup_delay_steps=train_config.startup_delay_steps,
                init_fn=init_fn,
                summary_op=summary_op,
                number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
                save_summaries_secs=120,
                sync_optimizer=sync_optimizer,
                saver=saver)

if __name__ == "__main__":
    args = parse_args()

    # Load original model graph
    original_dg = tf.Graph()
    with original_dg.as_default():
        saver = tf.train.import_meta_graph(args.model_meta, clear_devices=True)
    # Load original model ckpt
    o_sess = tf.Session(graph=original_dg)
    saver.restore(o_sess, args.model_ckpt)

    # Prune model
    with open(args.np_wpath, "rb") as f:
        pruned_weights = pickle.load(f)
    with open(args.np_spath, "rb") as f:
        pruned_shape = pickle.load(f)
    fine_tune_pruned_model(args.train_config, args.train_dir, pruned_shape, pruned_weights, args.model_ckpt)



