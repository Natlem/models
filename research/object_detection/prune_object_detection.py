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
from object_detection.trainer import create_input_queue,_create_losses


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
import prune_L1

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)

pruning_methods = ["L1", "Oracle", "Entropy"]




def parse_args():
    parser = argparse.ArgumentParser(description="Generate pruned model WITHOUT fine-tuning")

    parser.add_argument('--model_ckpt', dest='model_ckpt', help='original ckpt of model')
    parser.add_argument('--model_meta', dest='model_meta', help='original meta of model')
    parser.add_argument('--train_config', dest='train_config', help='original train config')
    parser.add_argument('--prune_percent', dest='prune_percent', help='percent of pruning')
    parser.add_argument('--pruning_method', dest="pruning_method", choices=pruning_methods, help="Pruning methods")
    args = parser.parse_args()
    return args

def prune_model(o_sess: tf.Session, original_dg: tf.Graph, prune_method: str, prune_percent: int, arch: str) -> dict:
    if prune_method == "L1":
        pruned_weights, fe1Shape = prune_L1.prune_L1(o_sess, original_dg, prune_percent, arch)
    return pruned_weights, fe1Shape

if __name__ == "__main__":
    args = parse_args()

    #Load original model graph
    original_dg = tf.Graph()
    with original_dg.as_default():
        saver = tf.train.import_meta_graph(args.model_meta, clear_devices=True)
    #Load original model ckpt
    o_sess = tf.Session(graph=original_dg)
    saver.restore(o_sess, args.model_ckpt)

    #Prune model
    pruned_weights, fe1Shape = prune_model(o_sess, original_dg, args.pruning_method, 50, "101") #dict of pruned weights

    filename = "pruned_weight_" + args.pruning_method + ".np"
    with open(filename, "wb") as f:
        pickle.dump(pruned_weights, f)
    filename = "pruned_shape_" + args.pruning_method + ".np"
    with open(filename, "wb") as f:
        pickle.dump(fe1Shape, f)



