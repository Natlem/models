import os
import sys
import numpy as np
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-9.0/lib64/"
import tensorflow as tf

sys.path.append("..")
sys.path.append("../slim/")

import pruning_utils
from pruning_utils import BLOCK_CST, SHORTCUT_CST, CONV_CST, BETA_CST, GAMMA_CST, BN_CST, MOVING_MEAN_CST, MOVING_VAR_CST, CONV2D_CST, RELU_CST

NUM_BLOCKS_RESNET_V1 = 4

def sortFilters(filters: np.ndarray) -> np.ndarray:
    c = filters.reshape(-1, filters.shape[3])
    c = np.linalg.norm(c, ord=1, axis=0)
    d = np.zeros([2, c.shape[0]])
    i = np.argsort(c)
    d[0,:] = i
    d[1,:] = c[i]
    return d

def removePercentL1(filters: np.ndarray, PrunePercent: int) -> np.ndarray:
    #Remove filters from numpy
    numOfRemove = filters.shape[1] * PrunePercent / 100.
    numRemoved_1 = int(numOfRemove)
    newFilters = filters[:,numRemoved_1:].copy()
    return newFilters

def sortRecoverFilters(l1Array: np.ndarray, filters: np.ndarray) -> np.ndarray:
    #Recover the order of the pruned filters
    b = np.argsort(l1Array[0, :])
    c = l1Array[:,b]
    c = c[0,:].astype(int)
    shape = np.asarray(filters.shape)
    shape[0] = c.shape[0]
    newFilters = np.zeros(shape)
    newFilters = filters[:,:,:,c]
    return newFilters

def pruneConvPercentL1(tensorWeights: np.ndarray, PrunePercent: int) -> (np.ndarray, list):
    #Prune out channels of convolutional layers
    d = sortFilters(tensorWeights)
    e = removePercentL1(d, PrunePercent)
    f = sortRecoverFilters(e, tensorWeights.copy())
    rindexes = e[0,:].astype(int).copy()
    rindexes.sort()
    return f, rindexes

def prune_unit_L1(unitDict: dict, rindexes: list, prunePercent: int, bnDict: dict, shapeUnitDict: dict) -> (list, dict, dict):
    # prune conv1 and shortcut in channels
    rindexesX = rindexes
    unitDict["conv1"] = pruning_utils.pruneConvInput(unitDict["conv1"], rindexes)
    if "shortcut" in unitDict.keys():
        unitDict["shortcut"] = pruning_utils.pruneConvInput(unitDict["shortcut"], rindexes)

    # prune conv1 out channels
    unitDict["conv1"], rindexes = pruneConvPercentL1(unitDict["conv1"], prunePercent[0])
    bnDict["conv1"] = pruning_utils.pruneBN(bnDict["conv1"], rindexes)

    # prune conv2 in channels
    unitDict["conv2"] = pruning_utils.pruneConvInput(unitDict["conv2"], rindexes)
    # prune conv2 out channels
    unitDict["conv2"], rindexes = pruneConvPercentL1(unitDict["conv2"], prunePercent[1])
    bnDict["conv2"] = pruning_utils.pruneBN(bnDict["conv2"], rindexes)
    # prune conv3 in channels
    unitDict["conv3"] = pruning_utils.pruneConvInput(unitDict["conv3"], rindexes)

    # prune shortcut and conv3 outchannels
    if "shortcut" in unitDict.keys():
        unitDict["shortcut"], rindexes = pruneConvPercentL1(unitDict["shortcut"], prunePercent[2])
        bnDict["shortcut"] = pruning_utils.pruneBN(bnDict["shortcut"], rindexes)
        unitDict["conv3"], rindexes = pruning_utils.pruneConvWithIndexes(unitDict["conv3"], rindexes)
    else:
        unitDict["conv3"], rindexes = pruning_utils.pruneConvWithIndexes(unitDict["conv3"], rindexesX)

    bnDict["conv3"] = pruning_utils.pruneBN(bnDict["conv3"], rindexes)

    # handle only shape
    shapeUnitDict["conv1"] = unitDict["conv1"].shape
    # prune conv2 in channels
    shapeUnitDict["conv2"] = unitDict["conv2"].shape
    # prune conv2 out channels
    shapeUnitDict["conv3"] = unitDict["conv3"].shape
    if "shortcut" in unitDict.keys():
        shapeUnitDict["shortcut"] = unitDict["shortcut"].shape

    return rindexes, bnDict

def get_pruned_weights(detection_graph: tf.Graph, sess: tf.Session, allConvFE1: dict, allConvFE2: dict, allBnFE1: dict, arch: str):
    o_weights = pruning_utils.get_all_original_weights(detection_graph, sess)
    o_weights['FirstStageFeatureExtractor/resnet_v1_X/conv1/weights'] = allConvFE1['conv1']
    o_weights["FirstStageFeatureExtractor/resnet_v1_X/conv1/BatchNorm/beta"] = allBnFE1['conv1'][BETA_CST]
    o_weights["FirstStageFeatureExtractor/resnet_v1_X/conv1/BatchNorm/gamma"] = allBnFE1['conv1'][GAMMA_CST]
    o_weights["FirstStageFeatureExtractor/resnet_v1_X/conv1/BatchNorm/moving_mean"] = allBnFE1['conv1'][MOVING_MEAN_CST]
    o_weights["FirstStageFeatureExtractor/resnet_v1_X/conv1/BatchNorm/moving_variance"] = allBnFE1['conv1'][
        MOVING_VAR_CST]
    final_dict = {}

    for k, v in o_weights.items():
        ky = k.replace(arch, "X")
        final_dict[ky] = o_weights[k]

        if "FirstStageFeatureExtractor" in k:
            if k.find(BLOCK_CST) == -1:
                continue
            blockID, unitID, convID = pruning_utils.getBUC(k)

            if SHORTCUT_CST in k:
                convID = SHORTCUT_CST
            if "weights" in k:
                final_dict[ky] = allConvFE1[blockID][unitID][convID]
            else:
                if k.endswith(BETA_CST):
                    key = BETA_CST
                elif k.endswith(GAMMA_CST):
                    key = GAMMA_CST
                elif k.endswith(MOVING_MEAN_CST):
                    key = MOVING_MEAN_CST
                elif k.endswith(MOVING_VAR_CST):
                    key = MOVING_VAR_CST
                final_dict[ky] = allBnFE1[blockID][unitID][convID][key]

    final_dict["Conv/weights"] = allConvFE2["Conv/weights"]
    final_dict["SecondStageFeatureExtractor/resnet_v1_X/block4/unit_1/bottleneck_v1/conv1/weights"] = \
    allConvFE2["block4"]["unit_1"]["conv1"]
    final_dict["SecondStageFeatureExtractor/resnet_v1_X/block4/unit_1/bottleneck_v1/shortcut/weights"] = \
    allConvFE2["block4"]["unit_1"]["shortcut"]

    return final_dict

def prune_L1(o_sess: tf.Session, original_dg: tf.Graph, prune_percent: int, arch: str = "101") -> (dict, dict):
    allConvFE1, allConvFE1Shape  = pruning_utils.get_1st_feature_extractor_weights(original_dg, o_sess)
    allConvFE2 = pruning_utils.get_2nd_feature_extractor_weights(original_dg, o_sess)
    allBnFE1 = pruning_utils.get_1st_feature_extractor_bn(original_dg, o_sess)

    allConvFE1["conv1"], rindexes = pruneConvPercentL1(allConvFE1['conv1'], prune_percent)
    allConvFE1Shape["conv1"] = allConvFE1["conv1"].shape

    allBnFE1["conv1"] = pruning_utils.pruneBN(allBnFE1["conv1"], rindexes)
    prunePercents = [prune_percent, prune_percent, prune_percent]

    # Prune all blocks
    for i in range(NUM_BLOCKS_RESNET_V1):
        block = allConvFE1["block{}".format(i + 1)]
        for k, v in block.items():
            rindexes, allBnFE1["block{}".format(i + 1)][k]  = prune_unit_L1(v, rindexes, prunePercents,
                                                                           allBnFE1["block{}".format(i + 1)][k], allConvFE1Shape["block{}".format(i + 1)][k])

    # Prune next FE 1st conv
    inChannels = int((len(rindexes) + 1) / 2)
    allConvFE2["Conv/weights"] = pruning_utils.pruneConvInputKeep(pruning_utils.getWeightsByName(original_dg, "Conv/weights", o_sess), inChannels)
    allConvFE2["block4"]["unit_1"]["conv1"] = pruning_utils.pruneConvInputKeep(allConvFE2["block4"]["unit_1"]["conv1"], inChannels)
    allConvFE2["block4"]["unit_1"]["shortcut"] = pruning_utils.pruneConvInputKeep(allConvFE2["block4"]["unit_1"]["shortcut"],
                                                                    inChannels)
    all_weights = get_pruned_weights(original_dg, o_sess, allConvFE1, allConvFE2, allBnFE1, arch)
    return all_weights, allConvFE1Shape