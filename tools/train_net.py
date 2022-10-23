#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import os
import numpy as np
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast_sddp.models.losses as losses
import slowfast_sddp.models.optimizer as optim
import slowfast_sddp.utils.checkpoint as cu
import slowfast_sddp.utils.distributed as du
import slowfast_sddp.utils.logging as logging
import slowfast_sddp.utils.metrics as metrics
import slowfast_sddp.utils.misc as misc
import slowfast_sddp.visualization.tensorboard_vis as tb
from slowfast_sddp.datasets import loader
from slowfast_sddp.models import build_model
from slowfast_sddp.utils.meters import AVAMeter, TrainMeter, ValMeter, EpochTimer

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    logger.info("Data size: {}".format(data_size))
    debug_iters = 0

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            if cfg.MODEL.PASTA_CLASSIFIER:
                preds, preds_pasta = model(inputs, meta["boxes"])
            else:
                preds = model(inputs, meta["boxes"])
        else:
            preds = model(inputs)

        # Explicitly declare reduction to mean.
        if cfg.MODEL.LOSS_FUNC == 'focal':
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)
        else:
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        
        # Compute the loss.
        if cfg.MODEL.ACTION_LOSS:    
            if cfg.MODEL.LOSS_FUNC == 'focal':
                loss = loss_fun(preds, labels, alpha=cfg.TRAIN.FOCAL_ALPHA, gamma=cfg.TRAIN.FOCAL_GAMMA, reduction="mean")
            else:
                loss = loss_fun(preds, labels)
        else:
            loss = None

        if cfg.MODEL.PASTA_CLASSIFIER:
            for part_idx in cfg.MODEL.NON_FROZEN_PASTAS:
                if loss is None:
                    loss = loss_fun(preds_pasta[part_idx], meta['pasta_label_arrs'][part_idx])
                else:
                    loss = loss + loss_fun(preds_pasta[part_idx], meta['pasta_label_arrs'][part_idx])

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        if cfg.DEBUG_MODE:
            debug_iters += 1
            if debug_iters != 0 and debug_iters % cfg.NUM_GPUS == 0:
                break
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    val_data_size = len(val_loader)
    logger.info("Data size: {}".format(val_data_size))
    debug_iters = 0

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            if cfg.MODEL.PASTA_CLASSIFIER:
                preds, preds_pasta = model(inputs, meta["boxes"])
            else:
                preds = model(inputs, meta["boxes"])
                preds_pasta = None
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]
            
            if cfg.MODEL.PASTA_CLASSIFIER:
                preds_pasta = torch.cat(preds_pasta, dim=1)

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()
                if cfg.MODEL.PASTA_CLASSIFIER:
                    preds_pasta = preds_pasta.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)
                if cfg.MODEL.PASTA_CLASSIFIER:
                    preds_pasta = torch.cat(du.all_gather_unaligned(preds_pasta), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata, preds_pasta)

        else:
            preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()
        if cfg.DEBUG_MODE:
            debug_iters += 1
            if debug_iters != 0 and debug_iters % cfg.NUM_GPUS == 0:
                break
    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/action_map": val_meter.full_map}, global_step=cur_epoch
            )
            if cfg.MODEL.PASTA_CLASSIFIER:
                for part_idx in range(len(cfg.AVA.PASTA_PART_NAMES)):
                    writer.add_scalars(
                        {f"Val/{cfg.AVA.PASTA_PART_NAMES[part_idx]}_map": val_meter.pasta_per_part_map[part_idx]}, global_step=cur_epoch
                    )
                writer.add_scalars(
                    {"Val/pasta_part_mean_ap": np.nanmean(val_meter.pasta_per_part_map)}, global_step=cur_epoch
                )
                writer.add_scalars(
                    {"Val/pasta_state_mean_ap": np.nanmean(val_meter.pasta_map)}, global_step=cur_epoch
                )
                
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Set up environment.
    du.init_distributed_training(cfg)
    
    # Make output dir and dump config file.
    if du.is_master_proc(du.get_world_size()):
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(cfg.OUTPUT_DIR, "configs.yaml"), 'w') as f:
            f.write(cfg.dump())

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR, shard_id=cfg.SHARD_ID)

    # Print config.
    logger.info("Train with config:")
    logger.info('\n'+str(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    if not cfg.TRAIN.VAL_ONLY:
        for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch, None)

            # Shuffle the dataset.
            loader.shuffle_dataset(train_loader, cur_epoch)

            # Train for one epoch.
            epoch_timer.epoch_tic()
            train_epoch(
                train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer
            )
            epoch_timer.epoch_toc()
            logger.info(
                f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
                f"from {start_epoch} to {cur_epoch} take "
                f"{epoch_timer.avg_epoch_time():.2f}s in average and "
                f"{epoch_timer.median_epoch_time():.2f}s in median."
            )
            logger.info(
                f"For epoch {cur_epoch}, each iteraction takes "
                f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
                f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
                f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
            )
            
            is_checkp_epoch = cu.is_checkpoint_epoch(cfg, cur_epoch, None)
            
            # Compute precise BN stats.
            if (
                (is_checkp_epoch or is_eval_epoch)
                and cfg.BN.USE_PRECISE_STATS
                and len(get_bn_modules(model)) > 0
            ):
                calculate_and_update_precise_bn(
                    precise_bn_loader,
                    model,
                    min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                    cfg.NUM_GPUS > 0,
                )
            _ = misc.aggregate_sub_bn_stats(model)

            # Save a checkpoint.
            if is_checkp_epoch:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
            # Evaluate the model on validation set.
            if is_eval_epoch:
                eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
    else:
        eval_epoch(val_loader, model, val_meter, cfg.SOLVER.MAX_EPOCH, cfg, writer)

    if writer is not None:
        writer.close()
