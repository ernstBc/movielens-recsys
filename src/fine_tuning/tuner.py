import os
from src.logger import logging as l
logging = l.getLogger(__name__)

import optuna
from typing import Literal, Tuple
from src.train.trainer import Trainer
from src.data.wrappers import UserItemDataSampling, AutoencoderSampling
from src.utils.utils import read_json, write_json


class Tuner:
    def __init__(self, 
                 study_name:str, 
                 n_trials:int,
                 model_type:str,
                 dataset_type:Literal['autoencoder', 'user_item', 'negative_sampling'],
                 data_dir_config:dict,
                 fine_tuning_config:dict,
                 model_kwargs:dict,
                 hyperparams_kwargs:dict,
                 trainer_kwargs:dict,
                 storage:bool=False):


        self.model_type = model_type
        self.dataset_type = dataset_type
        self.model_kwargs = model_kwargs
        self.data_dir_config = data_dir_config
        self.fine_tuning_config = fine_tuning_config
        self.hyperparams_kwargs = hyperparams_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.n_trials = n_trials


        if storage:
            STORAGE = self.data_dir_config['REGISTRY']['storage']
        else:
            STORAGE = None

        self.study = optuna.create_study(study_name=study_name,
                                         storage=STORAGE,
                                         load_if_exists=True)


    def _objective(self, trial, dataloder:UserItemDataSampling|AutoencoderSampling, max_epochs:int):
        def get_params(trial, PARAMS_TO_TUNE:str):
            params_finetuning = {}
            for param_name, param_config in self.fine_tuning_config[PARAMS_TO_TUNE.upper()].items():
                if param_config['type'] == 'categorical':
                            params_finetuning[param_name] = trial.suggest_categorical(param_name, param_config['values'])
                elif param_config['type'] == 'int':
                            params_finetuning[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                            params_finetuning[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config['logscale'] )
                elif param_config['type'] == 'list_int':
                            params_finetuning[param_name] = [trial.suggest_int(f"{param_name}_{i+1}", param_config['low'], param_config['high'], step=param_config['step'])
                                                             for i in range(
                                                                            trial.suggest_int(f"{param_name}_nlist", 
                                                                            param_config['min_elements'], 
                                                                            param_config['max_elements'])
                                                                            )
                                                        ]
                else:
                            raise ValueError(f"Unsupported parameter type: ;{param_config['type']};")
            return params_finetuning
            
        # Model Params Grid
        model_params = get_params(trial, self.model_type)

        # Hypermeters Grid
        hyper_params = get_params(trial, 'HYPERPARAMS')

        # Dataset hyperparams Grid JUST BATCH SIZE PARAMETER ALLOWED
        dataset_params = get_params(trial, 'DATASET')
        BATCH_SIZE = dataset_params['batch_size']

        # update the batch_size and stop processing the data after the first trial
        dl = dataloder
        dl.batch_size = BATCH_SIZE
        if trial.number == 1: 
               dl.force_process = False #type ignore 
               dl.process_data = False

        # update the model and hyperparams dicts with the current finetuning run
        model_args = self.model_kwargs | model_params
        hyperparams_args = self.hyperparams_kwargs | hyper_params

        # get a trainer 
        trainer = Trainer(
                          max_epochs=max_epochs,
                          model_type=self.model_type,
                          model_kwargs=model_args,
                          hyperparams_kwargs=hyperparams_args,
                          **self.trainer_kwargs)

        model = trainer.get_model()

        # train the model with new hyperparams
        results = trainer.train(model=model, dataloader=dl)

        # Log tuning metrics
        trial.set_user_attr("model_config", model_params)
        trial.set_user_attr("hyperparams_config", hyper_params)
        trial.set_user_attr('dataset_config', dataset_params)
        trial.set_user_attr('metrics', {k: v.item() for k, v in results.items() if isinstance(v, (int, float)) or hasattr(v, 'item')})
               
        # get the results
        val_loss = results['val_loss'].item()

        return val_loss


    def search_params(self, dataloader,max_epochs:int):
        self.study.optimize(
            lambda trial: self._objective(trial, dataloader, max_epochs), 
            n_trials=self.n_trials, 
            show_progress_bar=True)


    def get_best_params(self) -> Tuple[dict, dict, dict]:
           best_trial = self.study.best_trial
           model_config = best_trial.user_attrs['model_config']
           hyperparms_config = best_trial.user_attrs['hyperparams_config']
           dataset_config = best_trial.user_attrs["dataset_config"]

           return (model_config, hyperparms_config, dataset_config)


    def get_best_metrics(self) -> dict:
          best_trial = self.study.best_trial
          metrics = best_trial.user_attrs['metrics']
          return metrics


    def write_best_params_to_file(self, params:dict):
          filepath = os.path.join(self.data_dir_config['REGISTRY']['best_models_params'], f"{self.model_type}.json")
          write_json(params, filepath)
          

    def check_and_save_best_params_(self, candidate_best_params:dict):
        """Checks whether the candidate params are better than the previous best params."""
        best_current_params_path = os.path.join(self.data_dir_config['REGISTRY']['best_models_params'], f'{self.model_type.lower()}.json')

        if not os.path.exists(best_current_params_path):
            self.write_best_params_to_file(candidate_best_params)
            return True
        
        best_current_params = read_json(best_current_params_path)
        best_current_metric = best_current_params['val_loss']
        best_candidate_metric = candidate_best_params['val_loss']

        better = best_candidate_metric < best_current_metric

        if better:
              self.write_best_params_to_file(candidate_best_params)

        return better    