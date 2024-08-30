import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import lightning.pytorch as pl
import torch
from omegaconf import  OmegaConf
from typing import Any
from lightning.pytorch.utilities import grad_norm


from caduceus.modeling_caduceus import CaduceusForMaskedLM
from caduceus.configuration_caduceus import CaduceusConfig
import wandb

import src.utils as utils
import src.utils.train
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks


from src.tasks.metrics import cross_entropy
from src.tasks.torchmetrics import Perplexity,NumTokens

log = src.utils.train.get_logger(__name__)



class LitCaduceus(pl.LightningModule):
    def __init__(self,
                 config):
    # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        self.save_hyperparameters(config)

        # Check hparams
        #self._check_config()
        # To be set in `setup`
        self.model = None
        self.setup()

        self._state = None
        self.val_loader_names, self.test_loader_names = None, None
        self.validation_step_outputs = []
    def setup(self,stage=None):

        OmegaConf.update(
                self.hparams.model.config, "complement_map", {0: 0,1: 1,2: 2,3: 3,4: 4,5: 5,6: 6,7: 10,8: 9,9: 8,10: 7,11: 11 }, force_add=True)

        self.model = CaduceusForMaskedLM(config=CaduceusConfig(**OmegaConf.to_container(self.hparams.model.config,resolve=True)))
        self.loss = cross_entropy
        self.val_loss = cross_entropy
        self.perplexity = Perplexity()
        self.num_tokens = NumTokens()
    
    def _common_step(self, batch, batch_idx,mode = "train"):
        x, y = batch
        logits = self.model(x).logits

        if mode == "train":
            loss = self.loss(logits, y,ignore_index=4)
        else:
            loss = self.val_loss(logits, y,ignore_index=4)
        self.perplexity(logits, y,loss)
        self.num_tokens(logits, y)

        self.log_dict({f"{mode}/perplexity":self.perplexity,
                       f"{mode}/num_tokens":self.num_tokens},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True)

        return loss,logits

    def training_step(self, batch, batch_idx,dataloader_idx=0):
        loss,logits =self._common_step(batch, batch_idx,mode="train")
        self.log("train/loss", loss,on_step=True,on_epoch=False,sync_dist=True,prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx,dataloader_idx=0):
        loss,logits =self._common_step(batch, batch_idx,mode="validation")
        self.validation_step_outputs.append(logits)
        self.log("validation/loss", loss,on_step=False,on_epoch=True,sync_dist=True,prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx,dataloader_idx=0):
        loss,logits =self._common_step(batch, batch_idx,mode="test")
        self.log("test/loss", loss,on_step=False,on_epoch=False,sync_dist=True,prog_bar=True)
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)
    
    def on_validation_epoch_end(self):
        flattened_logits = torch.flatten(torch.cat(self.validation_step_outputs))
        self.logger.experiment.log({"validation/logits":wandb.Histogram(flattened_logits.to("cpu")),
                                    "global_step":self.global_step})
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self) -> Any:
        
        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

        #del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups:", hps)  # TODO: log.info throws error because hps is list of dicts
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        # Layer Decay
        if self.hparams.train.layer_decay['_name_'] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers:
                    num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (
                        self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizers param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)
        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]
