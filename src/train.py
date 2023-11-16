"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 11 Nov, 2023
"""
import argparse

import config

from dataloader import BEDataModule


import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from rt1 import RT1CRAM

import sys

from torchinfo import summary

import wandb


parser = argparse.ArgumentParser()

parser.add_argument("--epochs", "-ep", type=int, default=config.EPOCHS)
parser.add_argument("--train_batch_size", "-trbs", type=int, default=config.BATCH_SIZE)
parser.add_argument("--test_batch_size", "-tbs", type=int, default=1)
parser.add_argument("--learning_rate", "-lr", type=float, default=config.LR)
parser.add_argument("--cnn_arch", "-a",
                    type=str, default="efficientnet_b3")
parser.add_argument("--freeze_cnn", "-free",
                    type=bool, default=True)

if __name__ == "__main__":
    
    args = parser.parse_args()
        
    # set seed
    _ = seed_everything(config.SEED)
    
    # build model
    rt1 = RT1CRAM(
        cnn_bacnbone=config.SELECTED_CNN_BACKBONE, 
        num_res_blocks=config.NUM_RES_BLOCKS,
        freeze_cnn_backbone=args.freeze_cnn,
        args=args
    ).cuda()
    # print(rt1)

    summary(model=rt1)
    
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
    
    # build data module
    dm = BEDataModule()
    
    # run trainer
    trainer.fit(model=rt1, datamodule=dm)
    
    wandb.finish()