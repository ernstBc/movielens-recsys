import optuna
from typing import Literal
from src.train.trainer import Trainer, ConfigManager
from src.data.wrappers import UserItemDataSampling, AutoencoderSampling

class Tuner:
    def __init__(self, 
                 study_name:str, 
                 n_trials:int,
                 model_type:str,
                 dataset_type:Literal['autoencoder', 'user_item', 'negative_sampling'],
                 fine_tuning_config:dict,
                 model_kwargs:dict,
                 hyperparams_kwargs:dict,
                 trainer_kwargs:dict,
                 storage:str|None = None,
                 pruner:bool=False):


        self.model_type = model_type
        self.dataset_type = dataset_type
        self.model_kwargs = model_kwargs
        self.fine_tuning_config = fine_tuning_config
        self.hyperparams_kwargs = hyperparams_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.n_trials = n_trials

        if storage is not None:
            if pruner:
                self.study = optuna.create_study(study_name=study_name, 
                                                 storage=storage, 
                                                 load_if_exists=True,
                                                 pruner=optuna.pruners.MedianPruner())
            else:
                self.study = optuna.create_study(study_name=study_name, 
                                                 storage=storage, 
                                                 load_if_exists=True)
        else:
            if pruner:
                self.study = optuna.create_study(study_name=study_name,
                                                 load_if_exists=True,
                                                 pruner=optuna.pruners.MedianPruner())
            self.study = optuna.create_study(study_name=study_name, load_if_exists=True)


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

        # update the batch_size
        dl = dataloder
        dl.batch_size = BATCH_SIZE

        # update the model and hyperparams dicts with the current finetuning run
        model_args = self.model_kwargs | model_params
        hyperparams_args = self.hyperparams_kwargs | hyper_params

        # get a trainer 
        trainer = Trainer(model_type=self.model_type,
                          model_kwargs=model_args,
                          hyperparams_kwargs=hyperparams_args,
                          **self.trainer_kwargs)

        model = trainer.get_model()

        # train the model with new hyperparams
        results = trainer.train(model=model, dataloader=dl)
               

        # get the results
        val_loss = results['val_loss'].item()

        return val_loss


    def search_params(self, dataloader,max_epochs:int):
        self.study.optimize(
            lambda trial: self._objective(trial, dataloader, max_epochs), 
            n_trials=self.n_trials, 
            show_progress_bar=True)


