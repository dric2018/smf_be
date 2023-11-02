
import config

from dataloader import BEDataModule


import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from rt1 import RT1

from torchinfo import summary


if __name__ == "__main__":
    
    # set seed
    _ = seed_everything(config.SEED)
    
    # build model
    rt1 = RT1(
        cnn_bacnbone="efficientnet_b3", 
        num_res_blocks=config.NUM_RES_BLOCKS,
        freeze_cnn_backbone=False
    ).cuda()
    # print(rt1)

    summary(model=rt1)
    
    # define loggers
    # tb_logger = TensorBoardLogger(save_dir="tb_logs", name="RT1")
    wandb_logger = WandbLogger(project='SMF-Be', log_model='all')
    
    
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
            metrics_format=".3f",
        ),
        leave=True
    )

    # configure trainer
    trainer = Trainer(
        enable_progress_bar=True,
        # deterministic=True, 
        min_epochs=2, 
        max_epochs=config.EPOCHS, 
        gradient_clip_val=config.GRAD_CLIP_VAL, 
        # fast_dev_run=True,
        callbacks=[
            progress_bar,
            # MyProgressBar()
        ],
        logger=wandb_logger
    )
    
    # build data module
    dm = BEDataModule()
    
    # run trainer
    trainer.fit(model=rt1, datamodule=dm)