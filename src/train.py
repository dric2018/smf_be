"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 3 Jan, 2024
"""
import argparse

import config

from dataloader import BEDataModule

from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

import numpy as np
import os

from pprint import pprint

import random

from rt1 import RTCRAM

import sys

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchinfo import summary

from utils.model_utils import run_experiment 

import wandb

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", "-ep", type=int, default=config.EPOCHS)
parser.add_argument("--train_batch_size", "-trbs", type=int, default=config.BATCH_SIZE)
parser.add_argument("--test_batch_size", "-tbs", type=int, default=1)
parser.add_argument("--learning_rate", "-lr", type=float, default=config.LR)
parser.add_argument("--cnn_arch", "-a", type=str, default="efficientnet_b3")
parser.add_argument("--freeze_cnn", "-free", type=bool, default=config.FREEZE_CNN)
parser.add_argument("--vanilla", "-pt", type=bool, default=True)

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    pprint(vars(args))
        
    # set seed
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # build model
    rt1 = RTCRAM(
        cnn_bacnbone=config.SELECTED_CNN_BACKBONE, 
        num_res_blocks=config.NUM_RES_BLOCKS,
        freeze_cnn_backbone=args.freeze_cnn,
        args=None
    ).cuda()
    # print(rt1)

    # summary(model=rt1)
    
    # build data module
    dm = BEDataModule()
    
    if args.vanilla:        
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=config.TGT_PAD_TOK_ID, 
            label_smoothing=config.LABEL_SMOOTHING
        )
        opt = getattr(torch.optim, config.OPTIMIZER)(
            params=[p for p in rt1.parameters() if p.requires_grad], 
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY
        )

        scheduler = getattr(lr_scheduler, config.LR_SCHEDULER["type"])(**config.LR_SCHEDULER["params"], optimizer=opt)
        
        pprint(opt)
        
        pprint("LR Scheduler: "+config.LR_SCHEDULER["type"])
        pprint("Log file: "+config.LOGGING_FILE)

        ##### init experiment
        run = wandb.init(
            dir='../',
            project='SMF-Be', 
            group="RT1-CRAM", 
            name="be_model", 
            reinit=True
        )
        
        with open(config.MODEL_LOGGING_FILE, "w") as arch_file:
            arch_file.write(str(summary(rt1)))
            arch_file.write("\n")
            arch_file.write(str(opt))
            arch_file.write("\n")
            # log to wandb

        with open(config.LOGGING_FILE, "a") as f:   
            f.write("*** New experiment ***\n")

        _ = run_experiment(
            model=rt1, 
            dm=dm, 
            opt=opt, 
            loss_fn=loss_fn,
            scheduler=scheduler
        )

        wandb.finish()
        
    else:
        # define loggers
        wandb.init(
            project=config.PROJECT_NAME, 
            group=config.GROUP_NAME, 
            name=config.RUN_NAME, 
            reinit=True
        )

        wandb_logger = WandbLogger(
            name=config.RUN_NAME,
            project=config.PROJECT_NAME, 
            log_model=True, 
            save_dir=config.LOGS_PATH,
            checkpoint_name=config.RUN_NAME,
            group=config.GROUP_NAME, 
            reinit=True
        )

        # callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode="min",
            dirpath=config.MODEL_PATH,
            filename='be_model',
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False,
            every_n_epochs=1
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # create your own theme!
        progress_bar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="#1A1717",
                # progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="#1A1717",
                time="#1A1717",
                processing_speed="#1A1717",
                metrics="#1A1717",
                metrics_text_delimiter="\n",
                metrics_format=".7f",
            ),
            leave=True
        )

        # configure trainer
        trainer = Trainer(
            enable_progress_bar=True,
            # deterministic=True, 
            min_epochs=2, 
            max_epochs=config.EPOCHS, 
            # gradient_clip_val=config.GRAD_CLIP_VAL, 
            # fast_dev_run=True,
            callbacks=[
                progress_bar,
                lr_monitor,
                # MyProgressBar()
            ],
            logger=wandb_logger
        )

        # run trainer
        trainer.fit(model=rt1, datamodule=dm)

        wandb.finish()
