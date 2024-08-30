import hydra
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functools import wraps
from typing import List, Sequence
import psutil

import wandb
from lightning.pytorch.loggers import WandbLogger
from pretraining.lit_model import LitCaduceus
from pretraining.dataset import LitHG38, LitDynamicMultiSpecies
import src.utils as utils
from src.utils import registry
from omegaconf import OmegaConf



def create_trainer(config, **kwargs):
    wandb_logger = WandbLogger(project=config.wandb.project, entity=config.wandb.entity,log_model=True)

    # Lightning callbacks
    callbacks= []
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))


    # Configure ddp automatically
    n_devices = config.trainer.get('devices', 1)
    if isinstance(n_devices, Sequence):  # trainer.devices could be [1, 3] for example
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get('strategy', None) is None:
        config.trainer.strategy = dict(
            _target_='lightning.pytorch.strategies.DDPStrategy',
            gradient_as_bucket_view=True,
        )
    # special processing for seqlen warmup scheduler (reload)
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=wandb_logger)
    return trainer

@hydra.main(version_base=None,config_path="../configs", config_name="config.yaml")
def main(config:OmegaConf):


    # Initialize wandb
    wandb.login(key = os.environ.get("AYMEN_WANDB_API_KEY"))

    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))

    #Clean up the configs and print them
    # Get the available memory
    available_memory = psutil.virtual_memory().available
    # Convert bytes to gigabytes
    available_memory_gb = available_memory / (1024 ** 3)
    print(f"Available RAM: {available_memory_gb:.2f} GB")
    
    config = utils.train.process_config(config)
    s3_multispecies_path = "s3://sf-interns-ab19d84c831b4ff6-inputs/data/full_multispecies/multi_species_full.zip"

    #Load the data
    dataset_cfg = config.get("dataset")
    my_dataset = LitHG38(dataset_cfg)
    
    my_dataset.prepare_data()
    my_dataset.setup()

    # Initialize the Lightning model
    lt_model = LitCaduceus(config=config)

    trainer = create_trainer(config)
    utils.train.print_config(config, resolve=True)
    trainer.fit(lt_model,my_dataset)


if __name__ == "__main__":
    main()