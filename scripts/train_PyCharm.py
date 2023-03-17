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
    # model_name = "ConvMLPAutoEncoder"
    # model_name = "Conv2DAutoEncoder"
    # model_name = "Conv1DAutoEncoder"
    # model_name = "AdvMLPEnsemble"
    # model_name = "AdvConvMLPEnsemble"
    # model_name = "AdvConv1DEnsemble"
    model_name = "AdvConv1DEnsemble2"

    # config = dict(
    #     input_dim=2,
    #     hidden_dim=64,
    #     bidirectional=True,
    #     decoder_num_layers=3,
    #     encoder_num_layers=1,
    #     lr=5e-4,
    #     input_dropout=0.5,
    # )
    #
    # config = dict(
    #     input_dim=301,
    #     z_dim=64,
    #     lr=2e-4,
    #     dropout=0.0,
    #     norm=True,
    #     activation="SiLU",
    #     # num_channels=64
    # )

    # AdvMLPEnsemble
    # config = dict(
    #     input_dim=301,
    #     lr=1e-3,
    #     num_layers=3,
    #     num_models=8,
    #     adv_weight=1.,
    #     z_dim=16,
    #     anneal_max=1,
    #     ens_dropout=0.,
    #     weight_decay=1e-4,
    # )

    # # Conv2DAutoEncoder
    # config = dict(
    #     # input_dim=(1, 1, 100, 140),
    #     plot_percent=0.1,
    #     flat_bottleneck=False,
    #     # num_models=3,
    # )

    # # AdvConvMLPEnsemble
    # config = dict(
    #     ens_models=5
    #     )

    # # Conv1DAutoEncoder
    # config = dict(
    #     num_layers=4,
    #     num_channels=256,
    #     z_dim=32,
    #     kernel1=11,
    #     kernel2=3,
    #     dropout=0.1,
    #     bottleneck=True,
    #     norm=True,
    #     activation="SiLU",
    #     plot_percent=0.1,
    # )

    # "AdvConv1DEnsemble2"
    config = dict(
        ens_models=3,
        kernel1=[3, 7, 11]
        # kernel1 = [11]
    )


    # # MLP/ConvAutoEncoder
    # config = dict(
    #     input_dim=140,
    #     encoder_layers=4,
    #     decoder_layers=6,
    #     num_channels=64,
    #     kernel_size=7,
    #     kernel_size2=3,
    #     lr=1e-4,
    #     dropout=0.0,
    #     plot_percent=0.1,
    #     norm=True,
    #     activation="SiLU",
    # )

    model_choice = models.get(model_name, None)
    if not model_choice:
        raise KeyError(f"{model_name} is not a valid model!")


    # decide to train on GPU or CPU based on availability or user specified
    if not torch.cuda.is_available():
        GPU = 0
    else:
        GPU = 1

    logger = WandbLogger(
        entity="aproppe",
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

    # data = g2DataModule('pcfs_g2_devset_tiny.h5', batch_size=128, window_size=1)
    data = g2DataModule('pcfs_g2_n2666.h5', batch_size=128, window_size=1)

    # 2D
    # data = g2DataModule('pcfs_g2_2d_n1500.h5', batch_size=64, window_size=1, as_2d=True)

    model = model_choice(**config)


    # logger.watch(model, log="all")
    trainer.fit(model, data)

    trainer.save_checkpoint(f"../models/{model_choice.__name__}.ckpt")
    # model_artifact = wandb.Artifact(model_choice.__name__, type="model")
    # model_artifact.add_dir("../models")
    # logger.experiment.log_artifact(model_artifact)