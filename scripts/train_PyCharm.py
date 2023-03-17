import wandb
import torch
import pytorch_lightning as pl
import warnings
import os
from pytorch_lightning.loggers import WandbLogger
from g2_pcfs.models.base import models
from g2_pcfs.pipeline.data import g2DataModule
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# warnings.filterwarnings("ignore")

""" Just for debugging models in PyCharm before uploading for training on Google Colab """

if __name__ == '__main__':
    model_name = "AdvConv1DEnsemble"

    """ Choose model configuration """
    # AdvConvMLPEnsemble
    config = dict(
        ens_models=5
        )

    model_choice = models.get(model_name, None)
    if not model_choice:
        raise KeyError(f"{model_name} is not a valid model!")

    # decide to train on GPU or CPU based on availability or user specified
    if not torch.cuda.is_available():
        GPU = 0
    else:
        GPU = 1

    """ Set up wandb logger and pl-lightning trainer """
    logger = WandbLogger(
        entity="your entity name",
        project=model_name,
        log_model=False,
        save_code=False,
        offline=False,
    )

    trainer = pl.Trainer(
        max_epochs=30,
        gpus=GPU,
        logger=logger,
        checkpoint_callback=False
    )

    """ Load dataset (you will need to make your own .h5 file using make_devset.py in data/raw/ """
    data = g2DataModule('pcfs_g2_n2666.h5', batch_size=128, window_size=1)

    # For 2D convolutional autoencoder models
    # data = g2DataModule('pcfs_g2_2d_n1500.h5', batch_size=64, window_size=1, as_2d=True)

    model = model_choice(**config)

    trainer.fit(model, data)

    trainer.save_checkpoint(f"../models/{model_choice.__name__}.ckpt")
    
    """ For saving model """
    # model_artifact = wandb.Artifact(model_choice.__name__, type="model")
    # model_artifact.add_dir("../models")
    # logger.experiment.log_artifact(model_artifact)
