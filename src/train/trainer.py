import os
import datetime 
import logging

import torch
import pytorch_lightning as pl
from typing import Any, Literal
from src.utils.utils import read_yaml
from src.models.autoencoders import AutoEncoder, DeepAutoEncoder
from src.models.matrix_factorization import MatrixFactorization, DeepMatrixFactorization
from src.models.wrappers import AutoencoderWrapper, MatrixFactorizationWrapper



class ConfigManager:
    def __init__(self, config_path:str, config_settings:str|None=None):
        config = read_yaml(config_path)
        if config_settings is not None:
            config = config[config_settings]
        self.config= config


    def __call__(self, **kwds: Any) -> dict:
        new_config = self.config.copy()
        for nk, nv in kwds.items():
            new_config[nk] = nv

        return new_config


class Trainer:
    def __init__(self, 
                 model_type:str,
                 max_epochs:int,
                 model_kwargs:dict, 
                 hyperparams_kwargs:dict, 
                 verbose:bool=True, 
                 sanity_check_steps:int=2, 
                 profiler:str|None=None,
                 save_model_path:str|None=None,
                 save_intermediate_ckpts:bool=True,
                 from_checkpoint:str|None=None
    ):

        self.save_model_path = save_model_path
        self.verbose = verbose
        self.sanity_check_steps = sanity_check_steps
        self.profiler = profiler
        self.max_epochs = max_epochs
        self.save_intermediate_ckpts = save_intermediate_ckpts
        self.from_checkpoint = from_checkpoint
        self.model_type = model_type.lower()
        self.model_config = model_kwargs
        self.hyperparams_config = hyperparams_kwargs

        self.trainer = self._get_trainer(max_epochs=max_epochs, 
                                         save_model=save_intermediate_ckpts,
                                         verbose=verbose,
                                         sanity_check_steps=sanity_check_steps,
                                         profiler=profiler)



    def train(self, model, dataloader):
        self.trainer.fit(model, dataloader)

        if self.save_model_path is not None:
            self.save_model(model, self.save_model_path)

        return self.trainer.logged_metrics


    def get_model(self, from_path:str|None=None):
        if from_path is not None:
             model = self.load_model(from_path)
        else:
             model = self._get_model()
        return model
    

    def evaluate(self, model, dataloader):
        eval_metrics = self.trainer.test(model, dataloader)
        return eval_metrics


    def _get_model(self):
        if self.model_type == 'autoencoder':
            model = AutoEncoder(**self.model_config)
            model_pl = AutoencoderWrapper(model, **self.hyperparams_config)
        elif self.model_type =='deep_autoencoder':
            model = DeepAutoEncoder(**self.model_config)
            model_pl = AutoencoderWrapper(model, **self.hyperparams_config)
        elif self.model_type =='matrix_factorization':
            model = MatrixFactorization(**self.model_config)
            model_pl = MatrixFactorizationWrapper(model, **self.hyperparams_config)
        elif self.model_type == 'deep_matrix_factorization':
            model = DeepMatrixFactorization(**self.model_config)
            model_pl = MatrixFactorizationWrapper(model, **self.hyperparams_config)
        else:
            raise ValueError("Invalid model type")
        
        return model_pl


    def save_model(self, model, save_model_path):
        now = int(datetime.datetime.now().timestamp())
        model_dir = os.path.join(
            save_model_path, 
            f'{self.model_type}_{now}.pt'
        )
        
        torch.save(model.state_dict(), model_dir)


    def load_model(self, model_path:str):
         model = self._get_model()
         model.load_state_dict(torch.load(model_path))
         return model


    def _get_trainer(self, max_epochs:int, save_model:bool, verbose:bool, sanity_check_steps:int=2, profiler:str|None=None):
        if verbose:
                logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

                trainer = pl.Trainer(
                            max_epochs=max_epochs, 
                            enable_checkpointing=save_model,     
                            enable_progress_bar=True,
                            enable_model_summary=True,
                            num_sanity_val_steps=sanity_check_steps,
                            profiler=profiler)
        else:
                # Silence console warnings and info logs from Lightning
                #logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
                logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
                trainer = pl.Trainer(
                    max_epochs=max_epochs, 
                    enable_checkpointing=save_model,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    num_sanity_val_steps=sanity_check_steps,
                    profiler=profiler)
                
        return trainer