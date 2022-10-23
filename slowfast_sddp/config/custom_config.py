#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    # _C.AVA.TEST_GT_BOX_LISTS = ["ava_val_v2.2.csv"]
    _C.AVA.PASTA_GROUNDTRUTH_FILE = "video_pasta_val.csv"
    _C.AVA.PASTA_LABEL_MAP_DIR = "pasta_pbtxts"
    _C.AVA.PASTA_LABEL_MAP_FILE = "part_state.pbtxt"
    _C.AVA.PASTA_PART_NAMES = ['foot', 'leg', 'hip', 'hand', 'arm', 'head']
    _C.AVA.VAL_NO_CENTRE_CROP = False
    # _C.AVA.PASTA_MAPPING = {'foot': 0, 'leg': 1, 'hip': 2, 'hand': 3, 'arm': 4, 'head': 5}
    _C.MODEL.PASTA_CLASSES = [15, 14, 5, 33, 7, 13]
    _C.MODEL.PASTA_CLASSIFIER = False
    _C.MODEL.PASTA_CLASSIFIER_MED_DIMS = [512, 2048, 1024, 1024, 1024, 1024]
    _C.MODEL.FROZEN_PASTAS = []
    _C.MODEL.FROZEN_BACKBONE = False
    _C.MODEL.FROZEN_ACTION_HEADER = False
    _C.MODEL.FROZEN_VERB_CLASSIFIER = False
    _C.MODEL.ACTION_LOSS = True
    _C.MODEL.K700_BACKBONE = False
    _C.MODEL.VERB_CLASSIFIER_MED_DIM = -1
    _C.MODEL.PASTA_PROJECTION = True
    _C.MODEL.PASTA_WTS = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                          1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                          1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 1, 0, 0, 1, 0, 0, 0]]
    _C.TRAIN.FOCAL_GAMMA = 2.0
    _C.TRAIN.FOCAL_ALPHA = -1.0
    _C.TRAIN.PER_GPU_BS = 4
    _C.TRAIN.VAL_ONLY = False
    _C.DEBUG_MODE = False
    pass
