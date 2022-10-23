#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry
import slowfast_sddp.utils.logging as logging
logger = logging.get_logger(__name__)

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    for name, p in model.named_parameters():
        if 'head' not in name and cfg.MODEL.FROZEN_BACKBONE:
            p.requires_grad = False
        else:
            p.requires_grad = True
    
    if cfg.MODEL.FROZEN_ACTION_HEADER:
        for p in model.head.projection.parameters():
            p.requires_grad = False
    else:
        for p in model.head.projection.parameters():
            p.requires_grad = True
    
    if cfg.MODEL.PASTA_CLASSIFIER:
        for frozen_pasta in cfg.MODEL.FROZEN_PASTAS:
            pasta_idx = cfg.AVA.PASTA_PART_NAMES.index(frozen_pasta)
            for p in model.head.pasta_linear_classifiers[pasta_idx].parameters():
                p.requires_grad = False
                
        if cfg.MODEL.FROZEN_VERB_CLASSIFIER:
            for p in model.head.pasta_projection.parameters():
                p.requires_grad = False

    logger.info("Gradient States:")
    for name, p in model.named_parameters():
        logger.info("name: {}, shape: {}, grad state: {}".format(name, p.shape, p.requires_grad))

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    return model
