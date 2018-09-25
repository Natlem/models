import os
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-9.0/lib64/"
import sys
import tensorflow as tf
sys.path.append("..")
sys.path.append("../slim/")

BLOCK_CST = "block"
UNIT_CST = "unit"
SHORTCUT_CST = 'shortcut'
CONV_CST = "conv"
BETA_CST = "beta"
GAMMA_CST = "gamma"
BN_CST = "BatchNorm"
MOVING_MEAN_CST = "moving_mean"
MOVING_VAR_CST = "moving_variance"
CONV2D_CST = "Conv2D"
RELU_CST = "Relu"


def get_all_original_weights(detection_graph: tf.Tensor, sess: tf.Session) -> dict:
    o_weights = {}
    for n in detection_graph.as_graph_def().node:
        if n.name.endswith('weights'):
            o_weights[n.name] = getWeightsByName(detection_graph, n.name, sess)
        if n.name.endswith(BETA_CST):
            o_weights[n.name] = getWeightsByName(detection_graph, n.name, sess)
        elif n.name.endswith(GAMMA_CST):
            o_weights[n.name] = getWeightsByName(detection_graph, n.name, sess)
        elif n.name.endswith(MOVING_MEAN_CST):
            o_weights[n.name] = getWeightsByName(detection_graph, n.name, sess)
        elif n.name.endswith(MOVING_VAR_CST):
            o_weights[n.name] = getWeightsByName(detection_graph, n.name, sess)

    return o_weights

def pruneConvWithIndexes(tensorNP: tf.Tensor, rindexes: list) -> (tf.Tensor, list):
    #Prune out channels of convolutional layers using indexes
    filters = tensorNP[:,:,:,rindexes].copy()
    return filters, rindexes

def pruneConvInput(tensorNP: tf.Tensor, rindexes: list =None) -> tf.Tensor:
    #Prune input channels of everything
    if not rindexes is None:
        c = tensorNP[:,:,rindexes,:].copy()
    else:
        c = tensorNP.copy()
    return c

def pruneConvInputKeep(tensorNP: tf.Tensor, keep: int) -> tf.Tensor:
    #Prune input channels of everything
    c = tensorNP[:,:,0:keep,:].copy()
    return c

def getBUC(name: str) -> (str, str, str):
    blockID = name[name.find(BLOCK_CST): name.find(BLOCK_CST) + len(BLOCK_CST) + 1]
    unitID = name[name.find(UNIT_CST): name.find(UNIT_CST) + len(UNIT_CST) + 3]
    if unitID[-1] == "/":
        unitID = unitID[:-1]
    convID = name[name.find(CONV_CST): name.find(CONV_CST) + len(CONV_CST) + 1]
    return blockID, unitID, convID

def pruneBN(bnDict: dict, rindexes: list) -> dict:
    bnDict[GAMMA_CST] = bnDict[GAMMA_CST][rindexes]
    bnDict[BETA_CST] = bnDict[BETA_CST][rindexes]
    bnDict[MOVING_MEAN_CST] = bnDict[MOVING_MEAN_CST][rindexes]
    bnDict[MOVING_VAR_CST] = bnDict[MOVING_VAR_CST][rindexes]
    return bnDict

def getTensor(graph: tf.Graph, tensor_name: str) -> tf.Tensor:
    t = graph.get_tensor_by_name(tensor_name + ":0")
    return t
def getWeightsByName(graph: tf.Graph, tensor_name: str, sess: tf.Session) -> tf.Tensor:
    t = getTensor(graph, tensor_name)
    res = sess.run(t)
    return res
def getWeightsByTensor(sess: tf.Session, tensor: tf.Tensor) -> tf.Tensor:
    res = sess.run(tensor)
    return res
def get_1st_feature_extractor_tensors(detection_graph: tf.Graph) -> (dict, dict):
    allDict = {}
    allDictShape = {}
    currentBlockID = 0
    currentUnitID = 0
    convNum = 1

    for i, n in enumerate(detection_graph.as_graph_def().node):
        if "FirstStageFeatureExtractor" in n.name:
            if (CONV_CST in n.name.lower() or SHORTCUT_CST in n.name) and n.name.endswith('weights'):
                blockFind = n.name.find(BLOCK_CST)
                tensor = getTensor(detection_graph, n.name)
                if blockFind == -1:
                    allDict[CONV_CST + str(convNum)] = tensor
                    allDictShape[CONV_CST + str(convNum)] = tensor.shape
                else:
                    blockID, unitID, convID = getBUC(n.name)
                    if blockID != currentBlockID:
                        allDict[blockID] = {}
                        allDictShape[blockID] = {}
                        currentBlockID = blockID
                        currentUnitID = 0
                    if unitID != currentUnitID:
                        allDict[blockID][unitID] = {}
                        allDictShape[blockID][unitID] = {}
                        currentUnitID = unitID
                    if SHORTCUT_CST in n.name:
                        convID = SHORTCUT_CST
                    allDict[blockID][unitID][convID] = tensor
                    allDictShape[blockID][unitID][convID] = tensor.shape
    return allDict, allDictShape

def get_2nd_feature_extractor_tensors(detection_graph: tf.Graph) -> dict:
    allDict = {}

    prevName = ""
    currentBlockID = 0
    currentUnitID = 0
    currentConvID = 0
    shortcut = False
    c2 = 0
    convNum = 1

    for i, n in enumerate(detection_graph.as_graph_def().node):
        if "SecondStageFeatureExtractor" in n.name:
            if (CONV_CST in n.name.lower() or SHORTCUT_CST in n.name) and n.name.endswith('weights'):
                blockFind = n.name.find(BLOCK_CST)
                tensor = getTensor(detection_graph, n.name)
                if blockFind == -1:
                    allDict[n.name] = tensor
                else:
                    blockID, unitID, convID = getBUC(n.name)
                    if blockID != currentBlockID:
                        allDict[blockID] = {}
                        currentBlockID = blockID
                        currentUnitID = 0
                    if unitID != currentUnitID:
                        allDict[blockID][unitID] = {}
                        currentUnitID = unitID
                    if SHORTCUT_CST in n.name:
                        convID = SHORTCUT_CST
                    allDict[blockID][unitID][convID] = tensor
    return allDict
def get_1st_feature_extractor_weights(detection_graph: tf.Graph, sess: tf.Session) -> dict:
    allDict = {}
    allDictShape = {}
    currentBlockID = 0
    currentUnitID = 0
    convNum = 1

    for i, n in enumerate(detection_graph.as_graph_def().node):
        if "FirstStageFeatureExtractor" in n.name:
            if (CONV_CST in n.name.lower() or SHORTCUT_CST in n.name) and n.name.endswith('weights'):
                blockFind = n.name.find(BLOCK_CST)
                tensor = getTensor(detection_graph, n.name)
                weights = getWeightsByTensor(sess, tensor)
                if blockFind == -1:
                    allDict[CONV_CST + str(convNum)] = weights
                    allDictShape[CONV_CST + str(convNum)] = weights.shape
                else:
                    blockID, unitID, convID = getBUC(n.name)
                    if blockID != currentBlockID:
                        allDict[blockID] = {}
                        allDictShape[blockID] = {}
                        currentBlockID = blockID
                        currentUnitID = 0
                    if unitID != currentUnitID:
                        allDict[blockID][unitID] = {}
                        allDictShape[blockID][unitID] = {}
                        currentUnitID = unitID
                    if SHORTCUT_CST in n.name:
                        convID = SHORTCUT_CST
                    allDict[blockID][unitID][convID] = weights
                    allDictShape[blockID][unitID][convID] = weights.shape
    return allDict, allDictShape
def get_2nd_feature_extractor_weights(detection_graph: tf.Graph, sess: tf.Session) -> dict:
    allDict = {}

    prevName = ""
    currentBlockID = 0
    currentUnitID = 0
    currentConvID = 0
    shortcut = False
    c2 = 0
    convNum = 1

    for i, n in enumerate(detection_graph.as_graph_def().node):
        if "SecondStageFeatureExtractor" in n.name:
            if (CONV_CST in n.name.lower() or SHORTCUT_CST in n.name) and n.name.endswith('weights'):
                blockFind = n.name.find(BLOCK_CST)
                tensor = getTensor(detection_graph, n.name)
                weights = getWeightsByTensor(sess, tensor)
                if blockFind == -1:
                    allDict[n.name] = weights
                else:
                    blockID, unitID, convID = getBUC(n.name)
                    if blockID != currentBlockID:
                        allDict[blockID] = {}
                        currentBlockID = blockID
                        currentUnitID = 0
                    if unitID != currentUnitID:
                        allDict[blockID][unitID] = {}
                        currentUnitID = unitID
                    if SHORTCUT_CST in n.name:
                        convID = SHORTCUT_CST
                    allDict[blockID][unitID][convID] = weights
    return allDict
def get_1st_feature_extractor_bn(detection_graph: tf.Graph, sess: tf.Session) -> dict:
    bnDict = {}
    allDict = {}

    prevName = ""
    currentBlockID = 0
    currentUnitID = 0
    currentConvID = 0
    shortcut = False
    c2 = 0
    convNum = 1

    for i, n in enumerate(detection_graph.as_graph_def().node):
        if BN_CST in n.name:
            if (CONV_CST in n.name.lower() or SHORTCUT_CST in n.name):
                if n.name.endswith(BETA_CST):
                    bnDict[n.name] = getWeightsByName(detection_graph, n.name, sess)
                elif n.name.endswith(GAMMA_CST):
                    bnDict[n.name] = getWeightsByName(detection_graph, n.name, sess)
                elif n.name.endswith(MOVING_MEAN_CST):
                    bnDict[n.name] = getWeightsByName(detection_graph, n.name, sess)
                elif n.name.endswith(MOVING_VAR_CST):
                    bnDict[n.name] = getWeightsByName(detection_graph, n.name, sess)
    for k, v in bnDict.items():
        blockFind = k.find(BLOCK_CST)
        if k.endswith(BETA_CST):
            key = BETA_CST
        elif k.endswith(GAMMA_CST):
            key = GAMMA_CST
        elif k.endswith(MOVING_MEAN_CST):
            key = MOVING_MEAN_CST
        elif k.endswith(MOVING_VAR_CST):
            key = MOVING_VAR_CST

        if blockFind == -1:
            if not CONV_CST + str(convNum) in allDict:
                allDict[CONV_CST + str(convNum)] = {}
            allDict[CONV_CST + str(convNum)][key] = v
        else:
            blockID, unitID, convID = getBUC(k)
            if blockID != currentBlockID:
                allDict[blockID] = {}
                currentBlockID = blockID
                currentUnitID = 0
            if unitID != currentUnitID:
                allDict[blockID][unitID] = {}
                currentUnitID = unitID
            if SHORTCUT_CST in k:
                convID = SHORTCUT_CST
            if not convID in allDict[blockID][unitID]:
                allDict[blockID][unitID][convID] = {}
            allDict[blockID][unitID][convID][key] = v

    return allDict