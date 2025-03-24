# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
TENSORBOARD = False

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import sys
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils
import torch.nn as nn
import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate_CTC, train_one_epoch, train_one_epoch_CTC
import pickle

if TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter

    run = SummaryWriter()
else:
    import wandb
os.environ["WANDB_SILENT"] = "true"


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument(
        "--coco_path", type=str, default="/comp_robot/cv_public_dataset/COCO2017/"
    )
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--pretrain_model_path", help="load from other checkpoint")
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--new_class_embedding", action="store_true")
    parser.add_argument("--smart_mapping", action="store_true")
    parser.add_argument("--resume_finetuning", action="store_true")
    parser.add_argument("--path_old_charset", type=str, default=None)
    parser.add_argument("--random_erasing", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")

    return parser


class WithReplacementRandomSampler(Sampler):
    """Samples elements randomly, with replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # generate samples of `len(data_source)` that are of value from `0` to `len(data_source)-1`
        samples = torch.LongTensor(len(self.data_source))
        samples.random_(0, len(self.data_source))
        return iter(samples)

    def __len__(self):
        return len(self.data_source)


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)

    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, "use_ema", None):
        args.use_ema = False
    if not getattr(args, "debug", None):
        args.debug = False
    if not hasattr(args, "mode_chr"):
        args.mode_chr = True
    if not getattr(args, "eval_epoch", None):
        args.eval_epoch = 1

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(
        output=os.path.join(args.output_dir, "info.txt"),
        distributed_rank=args.rank,
        color=False,
        name="detr",
    )
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + " ".join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info("world size: {}".format(args.world_size))
    logger.info("rank: {}".format(args.rank))
    logger.info("local_rank: {}".format(args.local_rank))
    logger.info("args: " + str(args) + "\n")

    if not TENSORBOARD:
        run = wandb.init(project="OCRDETR-general-CTC", config=args, mode="disabled")
        run.define_metric("*", step_metric="global_step")

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()

    # build model

    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("number of params:" + str(n_parameters))
    logger.info(
        "params:\n"
        + json.dumps(
            {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
            indent=2,
        )
    )

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    args.charset = dataset_train.charset
    if args.new_class_embedding and args.resume_finetuning:
        device = args.device

        features_dim = model.class_embed[0].weight.data.shape[1]
        new_charset_size = len(
            args.charset
        )  ## Size of the charset corresponding to the new dataset

        new_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )
        new_decoder_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )
        new_enc_out_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )

        if model.dec_pred_class_embed_share:
            class_embed_layerlist = [
                new_class_embed for i in range(model.transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(new_class_embed)
                for i in range(model.transformer.num_decoder_layers)
            ]
        new_class_embed = nn.ModuleList(class_embed_layerlist)

        if args.smart_mapping:
            if args.path_old_charset is not None:
                old_charset = pickle.load(open(args.path_old_charset, "rb"))
            else:
                with open(
                    os.path.join(
                        os.path.dirname(datasets.__file__), "default_charset.json"
                    ),
                    "r",
                ) as f:
                    old_charset = json.load(f)
            not_mapped = []
            possible_mapping = list(range(len(old_charset)))
            mapping = {}
            for i, char in enumerate(args.charset):
                if char in old_charset:
                    mapping[i] = old_charset.index(char)
                    possible_mapping.remove(mapping[i])
                else:
                    not_mapped.append(char)

            while len(possible_mapping) < len(not_mapped):
                possible_mapping.append(np.random.randint(0, len(old_charset)))
            possible_mapping = list(np.random.permutation(possible_mapping))

            for i, char in enumerate(args.charset):
                if char not in old_charset:
                    mapping[i] = possible_mapping[0]
                    possible_mapping.pop(0)

            assert len(mapping) == len(args.charset)

            for j in range(model.transformer.num_decoder_layers):
                for i in range(new_charset_size):
                    new_class_embed[j].weight.data[i, :] = model.class_embed[
                        j
                    ].weight.data[mapping[i], :]
                    new_class_embed[j].bias.data[i] = model.class_embed[j].bias.data[
                        mapping[i]
                    ]

                    new_decoder_class_embed.weight.data[i, :] = (
                        model.transformer.decoder.class_embed[
                            j
                        ].weight.data[mapping[i], :]
                    )
                    new_decoder_class_embed.bias.data[i] = (
                        model.transformer.decoder.class_embed[j].bias.data[mapping[i]]
                    )

                    new_enc_out_class_embed.weight.data[i, :] = (
                        model.transformer.enc_out_class_embed.weight.data[mapping[i], :]
                    )
                    new_enc_out_class_embed.bias.data[i] = (
                        model.transformer.enc_out_class_embed.bias.data[mapping[i]]
                    )
        if args.smart_mapping:
            model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
            model.class_embed = new_class_embed.to(device)
            model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
            model.label_enc = nn.Embedding(
                len(dataset_val.charset) + 1, features_dim
            ).to(device)
            parameters_to_optimize = (
                list(model.class_embed.parameters())
                + list(model.transformer.decoder.class_embed.parameters())
                + list(model.transformer.enc_out_class_embed.parameters())
            )
        else:
            model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
            model.class_embed = new_class_embed.to(device)
            # model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
            model.label_enc = nn.Embedding(
                len(dataset_val.charset) + 1, features_dim
            ).to(
                device
            )  ### This is used for the dn process but will not be used during the finetuning, we define it to avoid errors

            parameters_to_optimize = (
                list(model.class_embed.parameters())
                + list(model.transformer.decoder.class_embed.parameters())
                + list(model.transformer.enc_out_class_embed.parameters())
            )

        # optimizer = torch.optim.AdamW(parameters_to_optimize, lr=args.lr,weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    scheduler_constant = torch.optim.lr_scheduler.StepLR(
        optimizer, 1, gamma=1, last_epoch=-1
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_constant, scheduler_constant], milestones=[2]
    )
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")

        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(args.output_dir)

    if os.path.exists(os.path.join(args.output_dir, "checkpoint.pth")):
        args.resume = os.path.join(args.output_dir, "checkpoint.pth")
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        model_without_ddp.load_state_dict(checkpoint["model"])
        if args.use_ema:
            if "ema_model" in checkpoint:
                ema_m.module.load_state_dict(
                    utils.clean_state_dict(checkpoint["ema_model"])
                )
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)

    if args.new_class_embedding and not args.resume_finetuning:
        device = args.device

        features_dim = model.class_embed[0].weight.data.shape[1]
        new_charset_size = len(
            args.charset
        )  ## Size of the charset corresponding to the new dataset

        new_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )
        new_decoder_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )
        new_enc_out_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )

        if model.dec_pred_class_embed_share:
            class_embed_layerlist = [
                new_class_embed for i in range(model.transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(new_class_embed)
                for i in range(model.transformer.num_decoder_layers)
            ]
        new_class_embed = nn.ModuleList(class_embed_layerlist)

        if args.smart_mapping:
            if args.path_old_charset is not None:
                old_charset = pickle.load(open(args.path_old_charset, "rb"))
            else:
                with open(
                    os.path.join(
                        os.path.dirname(datasets.__file__), "default_charset.json"
                    ),
                    "r",
                ) as f:
                    old_charset = json.load(f)

            not_mapped = []
            possible_mapping = list(range(len(old_charset)))
            mapping = {}
            for i, char in enumerate(args.charset):
                if char in old_charset:
                    mapping[i] = old_charset.index(char)
                    possible_mapping.remove(mapping[i])
                else:
                    not_mapped.append(char)

            while len(possible_mapping) < len(not_mapped):
                possible_mapping.append(np.random.randint(0, len(old_charset)))
            possible_mapping = list(np.random.permutation(possible_mapping))

            for i, char in enumerate(args.charset):
                if char not in old_charset:
                    mapping[i] = possible_mapping[0]
                    possible_mapping.pop(0)

            assert len(mapping) == len(args.charset)
            for j in range(model.transformer.num_decoder_layers):
                for i in range(new_charset_size):
                    new_class_embed[j].weight.data[i, :] = model.class_embed[
                        j
                    ].weight.data[mapping[i], :]

                    new_class_embed[j].bias.data[i] = model.class_embed[j].bias.data[
                        mapping[i]
                    ]

                    new_decoder_class_embed.weight.data[i, :] = (
                        model.transformer.decoder.class_embed[
                            j
                        ].weight.data[mapping[i], :]
                    )
                    new_decoder_class_embed.bias.data[i] = (
                        model.transformer.decoder.class_embed[j].bias.data[mapping[i]]
                    )

                    new_enc_out_class_embed.weight.data[i, :] = (
                        model.transformer.enc_out_class_embed.weight.data[mapping[i], :]
                    )
                    new_enc_out_class_embed.bias.data[i] = (
                        model.transformer.enc_out_class_embed.bias.data[mapping[i]]
                    )
        if args.smart_mapping:
            model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
            model.class_embed = new_class_embed.to(device)
            model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
            model.label_enc = nn.Embedding(
                len(dataset_val.charset) + 1, features_dim
            ).to(device)
            parameters_to_optimize = (
                list(model.class_embed.parameters())
                + list(model.transformer.decoder.class_embed.parameters())
                + list(model.transformer.enc_out_class_embed.parameters())
            )

        else:
            model.class_embed = new_class_embed.to(device)
            model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
            model.label_enc = nn.Embedding(
                len(dataset_val.charset) + 1, features_dim
            ).to(device)

            parameters_to_optimize = (
                list(model.class_embed.parameters())
                + list(model.transformer.decoder.class_embed.parameters())
                + list(model.transformer.enc_out_class_embed.parameters())
            )

        optimizer = torch.optim.AdamW(
            parameters_to_optimize, lr=args.lr, weight_decay=args.weight_decay
        )

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
        args.start_epoch = checkpoint["epoch"] + 1
        from collections import OrderedDict

        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict(
            {
                k: v
                for k, v in utils.clean_state_dict(checkpoint).items()
                if check_keep(k, _ignorekeywordlist)
            }
        )
        for k, v in utils.clean_state_dict(checkpoint).items():
            if check_keep(k, _ignorekeywordlist) is False:
                print("Ignored: {}".format(k))

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if "ema_model" in checkpoint:
                ema_m.module.load_state_dict(
                    utils.clean_state_dict(checkpoint["ema_model"])
                )
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)

    if args.eval:
        run = wandb.init(project="OCRDETR-general-CTC", config=args, mode="disabled")

        run.define_metric("*", step_metric="global_step")
        os.environ["EVAL_FLAG"] = "TRUE"
        test_stats, coco_evaluator = evaluate_CTC(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
            wo_class_error=wo_class_error,
            args=args,
            run=run,
        )
        if args.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )

        log_stats = {**{f"test_{k}": v for k, v in test_stats.items()}}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    # model.class_embed = new_class_embed

    print("Start training")
    args.start_epoch = checkpoint["epoch"] + 1
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        try:
            dataset_train.generates_synthetic_data()
        except:
            pass

        train_stats = train_one_epoch_CTC(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            wo_class_error=wo_class_error,
            lr_scheduler=lr_scheduler,
            args=args,
            logger=(logger if args.save_log else None),
            ema_m=ema_m,
            run=run,
        )

        if epoch % args.eval_epoch == 0:
            args.num_classes = len(dataset_train.charset)
            __, criterion, postprocessors = build_model_main(args)
            test_stats, coco_evaluator = evaluate_CTC(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
                wo_class_error=wo_class_error,
                args=args,
                logger=(logger if args.save_log else None),
                run=run,
                epoch=epoch,
                mode_chr=args.mode_chr,
            )
        
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }

            if True:
                checkpoint_path = output_dir / "checkpoint_best_regular.pth"
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )
                ep_paras = {"epoch": epoch, "n_parameters": n_parameters}
                epoch_time = time.time() - epoch_start_time
                epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
                log_stats.update(ep_paras)
                log_stats["epoch_time"] = epoch_time_str

                if args.output_dir and utils.is_main_process():
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # for evaluation logs
                    if coco_evaluator is not None:
                        (output_dir / "eval").mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ["latest.pth"]
                            if epoch % 50 == 0:
                                filenames.append(f"{epoch:03}.pth")
                            for name in filenames:
                                torch.save(
                                    coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name,
                                )

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                weights = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                if args.use_ema:
                    weights.update(
                        {
                            "ema_model": ema_m.module.state_dict(),
                        }
                    )
                utils.save_on_master(weights, checkpoint_path)

        # eval ema
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate_CTC(
                ema_m.module,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
                wo_class_error=wo_class_error,
                args=args,
                logger=(logger if args.save_log else None),
            )
            log_stats.update({f"ema_test_{k}": v for k, v in ema_test_stats.items()})
            if True:
                checkpoint_path = output_dir / "checkpoint_best_ema.pth"
                utils.save_on_master(
                    {
                        "model": ema_m.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        ep_paras = {"epoch": epoch, "n_parameters": n_parameters}

        try:
            log_stats.update({"now_time": str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get("copyfilelist")
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove

        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finetuning", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
