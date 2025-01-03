from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from arguments import get_args

def main():
    wandb_logger = WandbLogger(project="multi_defendant_ljp_my", log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="", mode="max")