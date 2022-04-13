#!/usr/bin/env python
# coding: utf-8
"""
Usage:
python train_maml_cifar100.py \
--gpu_id 3 \
--max_meta_iter 1000  \
--k_shot 4 --n_way 4  \
--train_bs 32 --eval_bs 32 --max_iter_for_eval 10 \
--lr_meta 1e-3 --lr_task 0.1



"""
# # Train MAML in pytorch-lightning
# - 2022-01-13 (r)
import os, sys
from pathlib import Path
import argparse
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

# Callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def main(args):
    from reprlearn.utils.misc import info, now2str, today2str, get_next_version_path, n_iter_per_epoch
    from reprlearn.data.datamodules.kshot_datamodule import KShotDataModule

    # ## Set Path
    # - DATA_ROOT:
    #   - Use '/data/hayley-old/Tenanbaum2000/data' for MNIST, Mono-MNIST, Rotated-MNIST, Teapots
    #   - Use `/data/hayley-old/maptiles_v2' folder for Maptile dataset
    ROOT = Path('/data/hayley-old/Tenanbaum2000')


    # ### Init DataModule
    dataset_name = 'cifar100'
    data_root = Path(args.data_root)
    in_shape = args.in_shape #(32, 32, 3)

    max_meta_iter = args.max_meta_iter #10000

    k_shot = args.k_shot #4
    n_way = args.n_way #5
    num_tasks_per_iter_for_train = args.train_bs #16
    num_tasks_per_iter_for_eval = args.eval_bs #16
    max_iter_for_train = max_meta_iter  # must be >= max-meta-iter
    max_iter_for_eval = args.max_iter_for_eval #10  # totoal number of loss_q's to be averaged over is num_tasks_per_iter_for_eval * max_iter_for_eval

    dm_config = {
        'dataset_name': dataset_name,
        'data_root': data_root,
        'k_shot': k_shot,
        'n_way': n_way,
        'num_tasks_per_iter_for_train': num_tasks_per_iter_for_train,
        'max_iter_for_train': max_iter_for_train,
        'num_tasks_per_iter_for_eval': num_tasks_per_iter_for_eval,
        'max_iter_for_eval': max_iter_for_eval,
    }
    dm = KShotDataModule(**dm_config)
    print(dm.name)

    # ### Init pl.Module
    from reprlearn.models.plmodules.meta_learning import MAML
    model_kwargs = {
        'lr_meta': args.lr_meta,
        'lr_task': args.lr_task,
        'use_averaged_meta_loss': args.use_averaged_meta_loss,
        'num_inner_steps': args.num_inner_steps,
        'log_every': args.log_every,
    }
    # net = get_densenet(output_size=n_way)
    net = None
    model = MAML(
        in_shape=args.in_shape,
        k_shot=args.k_shot,
        n_way=args.n_way,
        net=net,
        **model_kwargs,
    )
    print('lr_meta, lr_task', model.lr_meta, model.lr_task)
    print('num_inner_steps: ', model.num_inner_steps)

    # ### Init pl.Trainer
    # Init. callbacks
    # Model Checkpoint criterion
    ckpt_metric, ckpt_mode = 'val/loss_q', 'min'
    # ckpt_metric, ckpt_mode = 'loss', 'min'

    ckpt_callback = ModelCheckpoint(
        monitor=ckpt_metric,
        mode=ckpt_mode,
        save_top_k=5,
    )

    stop_metric, stop_mode = 'val/loss_q', 'min'
    # stop_metric, stop_mode = 'loss', 'min'

    stop_patience = 100
    early_stopping_callback = EarlyStopping(
        monitor=stop_metric,
        patience=stop_patience,
        mode=stop_mode,

    )

    callbacks = [ckpt_callback]
    # callbacks = [ckpt_callback, early_stopping_callback]



    # Init. Tensorboard logger
    exp_name = f'{model.name}_{dm.name}'
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=f'{ROOT}/lightning_logs/{today2str()}',
        name=exp_name,
        default_hp_metric=False,  # todo: what is this param's effect?
    )

    log_dir = Path(tb_logger.log_dir)
    print(tb_logger.log_dir)

    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        print("Created: ", log_dir)



    # Init. pl.Trainer
    trainer_config = {
        'gpus': 1,
        'max_epochs': args.max_meta_iter,
        'progress_bar_refresh_rate': 0,
        'terminate_on_nan': True,
        'check_val_every_n_epoch': 1,
        'logger': tb_logger,
        'callbacks': callbacks,
    }
    # trainer = pl.Trainer(fast_dev_run=3) # for test/debug
    trainer = pl.Trainer(**trainer_config)  # for real training

    # Fit model
    trainer.fit(model, dm)
    print(f"Finished at ep {trainer.current_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False) # add_help=False is important!
    parser.add_argument('--gpu_id', type=str, required=True)

    parser.add_argument('--max_meta_iter', type=int, required=True,
                        help="Number of meta-updates (aka. max_epochs)")
    #  DataModule parameters ##todo: make to kshot datamoudle's add-args method
    parser.add_argument('--data_root', type=str, default='/data/hayley-old/Tenanbaum2000/data')
    parser.add_argument('--in_shape', nargs=3, type=int, default=[32, 32, 3])
    parser.add_argument('--k_shot', type=int, required=True)
    parser.add_argument('--n_way', type=int, required=True)

    parser.add_argument('--train_bs', type=int, required=True)
    parser.add_argument('--eval_bs',type=int, required=True)
    parser.add_argument('--max_iter_for_eval', type=int, required=True)

    # Model params # todo: move to module's add_args method
    parser.add_argument('--lr_meta', type=float, default=1e-3)
    parser.add_argument('--lr_task', type=float, default=0.1)
    parser.add_argument('--num_inner_steps', type=int, default=1)
    # -- whether to use meta-loss as the average of the sum of the loss_q's in the
    # -- batch of tasks
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--use_averaged_meta_loss', dest='use_averaged_meta_loss', action='store_true')
    group.add_argument('--no_use_averaged_meta_loss', dest='use_averaged_meta_loss', action='store_false')
    parser.set_defaults(use_averaged_meta_loss=True)

    parser.add_argument('--log_every', type=int, default=10)

    args = parser.parse_args()
    print("Final args: ")

    # ------------------------------------------------------------------------
    # Initialize model, datamodule, trainer using the parsered arg dict
    # ------------------------------------------------------------------------
    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    main(args)
